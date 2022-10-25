from QVSlayers_gh import *
from QVSinit_gh import *
from QVSutils_gh import *

import torch
import torch.nn as nn
import torch.optim as optim

import h5py
import numpy as np
import datetime
import random
from tabulate import tabulate
import cv2
import scipy.io as scio
import time
import shutil


logtime = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')

class QVS_model(nn.Module):
    def __init__(self):
        super(QVS_model, self).__init__()

        self.cov = Cov()
        self.cov2 = Cov_2()
        self.div = Div()

        self.linear1 = nn.Linear(2, 1)
        self.linear2 = nn.Linear(2, 1)
        self.linear3 = nn.Linear(2, 1)
        self.linear4 = nn.Linear(2, 1)
        self.linear5 = nn.Linear(2, 1)

        self.linearout = nn.Linear(5, 1)

    def forward(self, seq,qvs_idx,sum_idx):
        Vlen = seq.shape[0]
        x1 = (self.cov(seq.view(Vlen, Fdim), qvs_idx,sum_idx)).mean().unsqueeze(0)
        x2 = (self.div(seq.view(Vlen, Fdim), qvs_idx,sum_idx)).mean().unsqueeze(0)

        x = torch.cat((x1, x2))

        xa = self.linear1(x)
        xb = self.linear2(x)
        xc = self.linear3(x)
        xd = self.linear4(x)
        xe = self.linear5(x)

        x = torch.cat((torch.sqrt(xa), torch.sqrt(xb), torch.sqrt(xc), torch.sqrt(xd), torch.sqrt(xe)))

        y = self.linearout(x)
        return(y)

def projection2(x_in,k,x_proj):
    x_in_proj = x_in[x_proj]
    x_len = len(x_in_proj)
    x_sorted,_ = torch.sort(x_in_proj) # sort to a non-decreasing sequence
    x_sum = 0
    x_idx = 0
    for i in range(x_len - 1, -1, -1):  # \lumda value

        x_tmp = x_sorted[i]
        x_sum += x_tmp
        f_tmp = x_sum - (x_len - i) * x_tmp
        if f_tmp >= k:
            x_idx = i + 1
            break
    x_lumda = (torch.sum(x_sorted[x_idx:x_len]) - k) / (x_len - x_idx)
    x_out = x_in_proj - x_lumda
    x_out[:][x_out[:] < 0] = 0
    x_out[:][x_out[:] > 1] = 1

    x_in_out = torch.zeros(x_in.shape).cuda()
    x_in_out[x_proj] = x_out
    return (x_in_out)

def roundingDL2(infrnum,k):
    frnum = infrnum.clone().detach().squeeze()
    i = (frnum == 1).nonzero().numel()
    while i <k:
        p = torch.max(frnum[:][frnum[:] != 1])
        chgidx = (frnum == p).nonzero().squeeze()
        elnum = chgidx.numel()
        if i + elnum <= k:
            i += elnum
            frnum[chgidx] = 1
        else:
            i += 1
            frnum[chgidx[torch.randperm(elnum)[0]]] = 1

    frnum[frnum[:]<1] = 0
    return(frnum)

def gen_uframe_hist(filelink):
    frame_hist = []
    filenum = 0

    for file in os.listdir(filelink):
        filesf = file.split('.')[1]
        if filesf == 'jpg' or filesf == 'jpeg':
            filenum += 1
            frame = cv2.imread(filelink+file)
            fheight = frame.shape[0]
            fwidth = frame.shape[1]
            frameg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_hist.append(cv2.calcHist([frameg], [0], None, [16], [0, 255]) /fheight /fwidth)

    return (frame_hist)

def cal_cus_f1(vname,gensum,args,thre=0.5):
    thre = torch.tensor(thre,dtype = torch.float).cuda()

    vfile = 'datasets/Frames_sampled/' + vname + '.mat'
    vdata = scio.loadmat(vfile)['vidFrame']
    vdataf = vdata[0,0]['frames']
    totel = 1

    usumlink = 'datasets/OVP_YouTube_cmp/' + vname + '/user'
    nas = gensum.nonzero().numel()
    flist = gensum.nonzero().squeeze()
    if flist.dim()==0:
        flist.unsqueeze_(0)

    fms = []
    precs = []
    recs = []

    for i in range(5):
        nmas = 0
        selidx = []
        usumfile = usumlink+str(i+1)+'/'
        usum = torch.tensor(gen_uframe_hist(usumfile),dtype=torch.float).cuda()
        nus = usum.shape[0]

        for j in range(nas):
            mindist = 0.6
            frame = vdataf[0,flist[j]]['cdata']
            fheight = frame.shape[0]
            fwidth = frame.shape[1]
            frameg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            framehist = torch.tensor(cv2.calcHist([frameg], [0], None, [16], [0, 255]) /fheight /fwidth,dtype=torch.float).cuda()
            for k in range(nus):
                if k not in selidx:
                    mdistsum = torch.abs(framehist-usum[k]).sum()
                    mdist = mdistsum/totel
                    if mdist < mindist:
                        mindist = mdist
                        minidx = k
            if mindist <= thre:
                nmas += 1
                selidx.append(minidx)
        prec = nmas/nas
        rec = nmas/nus
        precs.append(prec)
        recs.append(rec)
        fms.append(0 if prec==0 and rec==0 else 2*prec*rec/(prec+rec))

    max_idx = np.argmax(fms)

    return (fms[max_idx], precs[max_idx], recs[max_idx])

def localsearch_gd_cond(model_qvs,InVideoP, qvsidx,eval_KFs,sumidx_in,sumidx_proj,args):
    # greedy, deterministic
    enum = sumidx_in.shape[0]
    projnum = sumidx_proj.shape[0]
    xprev = torch.zeros(enum, 1).cuda()
    yprev = torch.zeros(enum, 1).cuda()
    yprev[sumidx_proj] = 1

    xprobsprev = model_qvs(InVideoP, qvsidx,xprev)
    yprobsprev = model_qvs(InVideoP, qvsidx,yprev)
    if not args.cond:
        idxs = sumidx_in[sumidx_proj, 0].argsort(descending=True)
    else:
        idxs = torch.arange(projnum).cuda()
    
    k = 0

    for i in range(projnum):
        idxcurr = sumidx_proj[idxs[i]]
        xcurr = xprev.detach().clone()
        xcurr[idxcurr,0] = 1
        ycurr = yprev.detach().clone()
        ycurr[idxcurr,0] = 0

        xprobscurr = model_qvs(InVideoP, qvsidx,xcurr)
        yprobscurr = model_qvs(InVideoP, qvsidx, ycurr)
        a = xprobscurr - xprobsprev
        b = yprobscurr - yprobsprev

        if a > b:
            xprev = xcurr.detach().clone()
            xprobsprev = xprobscurr
            k += 1
        else:
            yprev = ycurr.detach().clone()
            yprobsprev = yprobscurr

        if k >= eval_KFs:
            break

    return(xcurr.squeeze())

def sim_proj(InVideoP, qvsidx):
    qvsidx2 = torch.nonzero(qvsidx.squeeze())
    if qvsidx2.dim() > 1:
        qvsidx2.squeeze_(1)
    qvsnorm = torch.norm(InVideoP[:, None] - InVideoP,dim=2)
    qvsnormsel = qvsnorm[:,qvsidx2] + 1e-8
    qvsnormavg = torch.mean(qvsnormsel,1)
    qvsnormmed = qvsnormavg.median()
    qvsnormavg[qvsnormavg > qvsnormmed] = 0
    qvsnormnum = torch.numel(torch.nonzero(qvsnormavg).squeeze())
    qvsnormavg[qvsnormavg == 0] = float('inf')

    return (torch.sort(qvsnormavg,descending=False)[1][0:qvsnormnum])

def evaluate_qvs_cond(model_qvs, dataset, test_keys, args,cond_flag,cond_idx=''):
    logger.info("==> Test sub")

    model_eval = QVS_model()
    model_eval = nn.DataParallel(model_eval).cuda()
    model_eval.module.load_state_dict(model_qvs.module.state_dict())
    model_eval.eval()

    if args.verbose: table = [["No.", "Video", "F-score", "Precision", "Recall"]]
    if args.save_results:
        h5_res = h5py.File(osp.join(args.save_dir, logtime + 'result.h5'), 'w')

    fms = []
    precs = []
    recs = []
    vnum = 0
    final_idx_all = []
    query_idx_all = []
    for key_idx, key in enumerate(test_keys):
        vnum += 1
        InVideo = dataset[key]['features'][...]
        vname = dataset[key]['video_name'][...]
        VideoLen = InVideo.shape[0]

        gtsummary = torch.from_numpy(dataset[key]['gtsummary'][...]).float().cuda()

        InVideoP = torch.from_numpy(InVideo).float().cuda()
        sumidx = torch.zeros(VideoLen,1).cuda()
        if cond_flag:
            eval_KFs = round(VideoLen * 0.02)
            eval_proj = round(VideoLen * 0.02)
            qvsidx = torch.zeros(VideoLen, 1).cuda()  # cond vsumm
            qvsidx[cond_idx] = 1
        else:
            if args.ftype == 'googlenet':
                eval_KFs = round(VideoLen * 0.05) # for googlenet feature
                eval_proj = round(VideoLen * 0.15) # for googlenet feature
                qvsidx = torch.ones(VideoLen, 1).cuda()  # generic vsumm
            elif args.ftype == 'color':
                eval_KFs = round(VideoLen * 0.1)  # for color feature
                eval_proj = round(VideoLen * 0.1)  # for color feature
                qvsidx = torch.ones(VideoLen, 1).cuda()  # generic vsumm
            else:
                logger.info("Feature type error.")

        logger.debug(sumidx.sum())
        logger.info("# {}th , # video key {}, # video name {}, # video length {}".format(vnum, key, vname,VideoLen))

        iter = args.eval_iter
        lr = args.eval_lr #learning rate
        logger.info("The learning rate is {}".format(lr))

        sumidx_tot = torch.zeros(VideoLen,1).cuda()
        sumidx_intm = torch.zeros(VideoLen, 4).cuda()
        probs_intm = torch.zeros(1, 4).cuda()

        gradtype = args.optim
        if gradtype == 'SGD':
            optim_eval = torch.optim.SGD([sumidx], lr=lr)
        elif gradtype == 'SGDM':
            optim_eval = torch.optim.SGD([sumidx], lr=lr, momentum=0.9)
        elif gradtype == 'AdaGrad':
            optim_eval = torch.optim.Adagrad([sumidx],lr=lr)
        elif gradtype == 'RMSProp':
            optim_eval = torch.optim.RMSprop([sumidx],lr=lr)
        elif gradtype == 'Adam':
            optim_eval = torch.optim.Adam([sumidx],lr=lr)
        else:
            logger.error("Optimization type error.")
            sys.exit(0)

        if cond_flag:
            sumidx_proj = sim_proj(InVideoP, qvsidx)
        else:
            sumidx_proj = torch.arange(VideoLen).cuda()
        sumidx[sumidx_proj] = eval_proj / torch.numel(sumidx_proj)
        for r in range(0,3,2):
            for i in range(iter):

                sumidx.requires_grad_()
                optim_eval.zero_grad()
                probs = model_eval(InVideoP, qvsidx,sumidx)*(-1)
                probs.backward()
                optim_eval.step()
                sumidx.requires_grad_(requires_grad=False)
                sumidx.data = projection2(sumidx, eval_proj,sumidx_proj)
                sumidx_tot += sumidx

                if i % 50 == 0:
                    logger.info("{}th iteration processed".format(i + 1))
                    logger.debug("The probes is: {}".format(probs))

            logger.info("Rounding starts")
            gtprobs = model_eval(InVideoP, qvsidx,gtsummary.unsqueeze(1))
            logger.info("GT probes: {}".format(gtprobs))
            logger.info("Final probes: {}".format(probs))
            logger.debug(sumidx_tot.sum())

            sumidx_avg = sumidx_tot / iter
            sumidx_intm[:,r+0] = sumidx_avg.squeeze()
            sumidx_ls = localsearch_gd_cond(model_eval, InVideoP, qvsidx, eval_KFs, sumidx_avg,sumidx_proj,args)  # cond vsumm
            sumidx_intm[:, r + 1] = sumidx_ls
            sumidx = projection2((1 - sumidx_avg), eval_proj,sumidx_proj)

        for n in range(4):
            sumidx_tmp = sumidx_intm[:,n].unsqueeze(1)
            probs_intm[0,n] = model_eval(InVideoP, qvsidx,sumidx_tmp)
        sumidx_final = sumidx_intm[:,torch.max(probs_intm,1)[1]]
        final_idx = roundingDL2(sumidx_final, eval_KFs)

        filelink = 'datasets/Oracle_' + args.metric + '.mat'
        oracle_record = scio.loadmat(filelink)['Oracle_record']
        samplenam = oracle_record[:, 0].astype(int)

        vfile = 'datasets/Frames_sampled/' + vname + '.mat'

        vdata = scio.loadmat(vfile)['vidFrame']
        vdataf = vdata[0, 0]['nrFramesTotal']
        final_idx_new = torch.zeros(int(vdataf)).cuda()
        qvsidx_new = torch.zeros(int(vdataf)).cuda()

        if args.ftype == 'color':
            samples = torch.from_numpy(np.array(oracle_record[np.where(samplenam == int(str(vname)[1:])), 2][0, 0],
                                                dtype=int)).squeeze().cuda()
            final_idx_new[torch.tensor(samples[torch.nonzero(final_idx)].squeeze() / 15 - 1,
                                       dtype=torch.long)] = 1  # for color features (861)
            final_idx_all.append(final_idx_new)
            fm, prec, rec = cal_cus_f1(vname, final_idx_new, args)  # for color features (861)
            qvsidx_new[torch.tensor(samples[torch.nonzero(qvsidx.squeeze())].squeeze()/15 - 1,dtype=torch.long)] = 1
            query_idx_all.append(qvsidx_new)
        else:
            fm, prec, rec = cal_cus_f1(vname, final_idx, args)
            final_idx_all.append(final_idx)
            query_idx_all.append(qvsidx.squeeze())

        fms.append(fm)
        precs.append(prec)
        recs.append(rec)

        if args.verbose:
            table.append([key_idx + 1, key, "{:.1%}".format(fm),"{:.1%}".format(prec),"{:.1%}".format(rec)])

        if args.save_results:
            h5_res.create_dataset(key + '/PGA_summary', data=sumidx_final.cpu()) #bf rounding
            h5_res.create_dataset(key + '/fm', data=fm)
            h5_res.create_dataset(key + '/prec', data=prec)
            h5_res.create_dataset(key + '/rec', data=rec)

    if args.verbose:
        logger.info(tabulate(table))

    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    mean_prec = np.mean(precs)
    mean_rec = np.mean(recs)
    logger.info("Average F-score {:.1%}, Precision {:.1%}, Recall {:.1%}: ".format(mean_fm,mean_prec,mean_rec))
    return mean_fm,final_idx_all,query_idx_all

def copy_frame_cond(in_idx,test_key,in_link,out_link,query=False):
    fignumorig = (in_idx + 1) * 15
    in_fignam = test_key + '_' + str(fignumorig) + '.jpg'
    if not query:
        out_fignam = test_key + '_' + str(in_idx) + '.jpg'
    else:
        out_fignam = 'query' + '_' + str(in_idx) + '.jpg'
    shutil.copyfile(in_link + in_fignam,out_link + out_fignam)
    return

def get_fig_cond(sum_idx_all,qry_idx_all,dataset,test_keys,args):
    fig_link = 'datasets/Frames/'

    vnum = len(test_keys)
    for i in range(vnum):
        keys = test_keys[i]
        test_key_in = dataset[keys]['video_name'][...]
        test_key = str(np.array(test_key_in, dtype=str))
        sum_idx = torch.nonzero(sum_idx_all[i]).squeeze().cpu().numpy()
        logger.debug(sum_idx) # for debug
        qry_idx = torch.nonzero(qry_idx_all[i]).squeeze().cpu().numpy()
        if np.size(qry_idx) == 1:
            qry_idx = np.reshape(qry_idx,(1))
        if np.size(sum_idx) == 1:
            sum_idx = np.reshape(sum_idx, (1))

        snum = np.size(sum_idx)
        qnum = len(qry_idx)
        output_link = 'QVSmodels/' + logtime + '/' + test_key +'/'
        if os.path.exists(output_link):
            shutil.rmtree(output_link)
        os.makedirs(output_link)
        input_link = fig_link + args.metric + '/' + test_key + '/'

        for j in range(snum):
            copy_frame_cond(sum_idx[j],test_key,input_link,output_link,0)

        for j in range(qnum):
            copy_frame_cond(qry_idx[j], test_key, input_link, output_link,1)

    return

def train_qvs(model, dataset, train_keys, num_train, optimizer, test_keys,args):

    start_time = time.time()

    vnum = 0
    epnum = 1
    last_epoch_time = start_time
    train_cont = True
    no_upd_cnt = 0
    prev_fm, curr_fm = 0, 0
    max_fm, max_ken = 0, 0
    # while(train_cont):
    while (epnum <= 10):

        model.train()
        logger.info("==> Train sub")

        idxs = np.arange(num_train)  # per training video
        np.random.shuffle(idxs)  # shuffle indices
        mbsize = args.mbsize
        for itmp in range(mbsize):
            for idx in idxs: #for each training video
                vnum += 1
                keys = train_keys[idx]
                InVideo = dataset[keys]['features'][...]
                VideoLen = InVideo.shape[0]
                train_KFs = round(VideoLen * 0.15)

                gtsummary = torch.from_numpy(dataset[keys]['gtsummary'][...]).float().cuda()

                logger.info("# {}th , # video name {}, # video length {}".format(vnum, keys,VideoLen))
                logger.debug('gt summary:', torch.nonzero(gtsummary).squeeze())

                InVideoP = torch.from_numpy(InVideo).float().cuda()

                iter = args.max_iter

                qvsidx = torch.ones(VideoLen, 1).cuda() # generic video summarization
                sumidx = torch.zeros(VideoLen, 1).cuda()
                sumidx[:] = train_KFs / VideoLen

                with torch.no_grad():
                    probs_gt = model(InVideoP,qvsidx,gtsummary.unsqueeze(1))
                for i in range(iter):
                    optimizer.zero_grad()
                    probs = model(InVideoP, qvsidx,sumidx)

                    loss = nn.L1Loss().cuda()
                    # loss = nn.MSELoss().cuda()
                    output = loss(probs, probs_gt)
                    output.backward()
                    optimizer.step()
                    with torch.no_grad():
                        for name, param in model.named_parameters():
                            param.clamp_(min=0)

                    logger.info("the {}th epoch, the {}th iteration, loss is {}".format(epnum, i, output))

        vnum = 0
        epoch_time = time.time()
        elapsed = str(datetime.timedelta(seconds=round(epoch_time - last_epoch_time)))
        logger.info("The {}th epoch takes time (h:m:s): {}".format(epnum, elapsed))

        model_save_path = osp.join(args.save_dir, 'model_epoch' + logtime + '-' + str(epnum) + '.pth.tar')

        save_checkpoint(model, model_save_path)
        logger.info("Model saved to {}".format(model_save_path))

        last_epoch_time = epoch_time

        start_time = time.time()
        prev_fm = curr_fm
        curr_fm,final_idx_all = evaluate_qvs_cond(model, dataset, test_keys, args, False)
        test_time = time.time()
        elapsed = str(datetime.timedelta(seconds=round(test_time - start_time)))
        logger.info("Testing Finished. Total elapsed time (h:m:s): {}".format(elapsed))

        if curr_fm > max_fm:
            max_fm = curr_fm

        if curr_fm <= prev_fm:
            no_upd_cnt += 1
        else:
            no_upd_cnt = 0
        if no_upd_cnt >= 5:
            train_cont = False
        epnum += 1
        logger.info('max fm: {}'.format(max_fm))

    train_time = time.time()
    elapsed = str(datetime.timedelta(seconds=round(train_time - start_time)))

    logger.info("Training Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    return

def main():
    args = get_arguments()
    if not args.mode==2:
        logfile = args.save_dir + logtime + '-log_train.txt'
    else:
        logfile = args.save_dir + logtime + '-log_test.txt'
    SetLogging(logfile)

    logger.info("==========\nArgs:{}\n==========".format(args))

    if torch.cuda.is_available():
        logger.info("Currently using GPU {}".format(args.gpu))

    dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())

    splits = ReadJson(args.split)
    assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
    split = splits[args.split_id]

    train_keys = split['train_keys']
    test_keys = split['test_keys']
    num_train = len(train_keys)
    num_test = len(test_keys)
    logger.info("# total videos {}. # train videos {}. # test videos {}".format(num_videos, num_train, num_test))

    run_mode = args.mode

    model_st = QVS_model()
    model_st = nn.DataParallel(model_st).cuda()
    model_st.apply(weights_init)

    model_qvs = QVS_model()
    model_qvs = nn.DataParallel(model_qvs).cuda()
    model_qvs.apply(weights_init)

    for name, param in model_qvs.named_parameters():  # debug
        if param.lt(0).nonzero().numel() > 0:
            logger.debug(name)

    if run_mode == 2: #evaluate

        if args.ftype == 'googlenet' and args.metric == 'OVP':
            model_qvs = torch.load('models/model_ovp-googlenet.pth.tar')  # ovp model for googlenet feature.
        elif args.ftype == 'googlenet' and args.metric == 'Youtube':
            model_qvs = torch.load('models/model_youtube-googlenet.pth.tar')  # youtube model for googlenet feature.
        elif args.ftype == 'color' and args.metric == 'OVP':
            model_qvs = torch.load('models/model_ovp-color.pth.tar')  # ovp model for color feature.
        elif args.ftype == 'color' and args.metric == 'Youtube':
            model_qvs = torch.load('models/model_youtube-color.pth.tar')  # youtube model for color feature.
        else:
            logger.info("Please specify dataset (OVP / Youtube) and feature type (googlenet / color).")

        model_qvs.eval()
        for name, param in model_qvs.named_parameters(): #debug
            if param.lt(0).nonzero().numel() > 0:
                # print(name)
                logger.debug(name)

        if not args.cond:
            curr_fm, final_idx_all, _ = evaluate_qvs_cond(model_qvs, dataset, test_keys, args, False) # generic vsumm
        else:
            curr_fm, final_idx_all, query_idx_all = evaluate_qvs_cond(model_qvs, dataset, test_keys,args,True,args.query)  # cond vsumm
            get_fig_cond(final_idx_all, query_idx_all, dataset, test_keys, args)  # cond vsumm

    elif run_mode == 1: #train qvs model
        optimizer_qvs = optim.Adagrad(model_qvs.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_qvs(model_qvs, dataset, train_keys, num_train, optimizer_qvs, test_keys,args)
    else:
        logger.info("run mode error.")

    dataset.close()
    return

if __name__ == '__main__':
    main()


