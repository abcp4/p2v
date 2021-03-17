import logging

def run(
    data,
    output = None,
    alg = None,
    l2 = 1,
    iteration = 10,
    total_samples = None,
    window = 5,
    noise_distribution = 0.75,
    negative = 5,
    workers = 1,
    p = 0.85,
    min_alpha = 0.0001,
    alpha = 0.025,
    undirected = True,
    log_file = None,
    force_shuffle = False,
    report_delay = 1,
):
    alg = data.alg if alg is None else alg


    if "n2v" in alg.lower() and p is None:
        raise ValueError("You must pass walking probability p")
    if "line" in alg.lower():
        p = 0
    if log_file:
        formatter = logging.Formatter('%(asctime)s %(message)s')
        filelogger = logging.getLogger()

        for h in filelogger.handlers:
            filelogger.removeHandler(h)

        filelogger.setLevel(logging.INFO)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        filelogger.addHandler(fh)

    import model as n2v_r

    if "line" in alg.lower() or "n2v" in alg.lower():
        w2v_ratio = 0
        n2v_ratio = 1
        shuffle = 1
    else:
        w2v_ratio = 1
        n2v_ratio = 0
        if force_shuffle:
            shuffle = 1
        else:
            shuffle = 0

    import numpy as np
    from collections import defaultdict
    model = n2v_r.paper2vec(
        data,
        w2v_ratio = w2v_ratio,
        d2v_ratio = 0,
        n2v_ratio = n2v_ratio,
        workers = workers,
        w2v_window = window,
        alpha = alpha,
        min_alpha = min_alpha,
        w2v_min_count = 0,
        negative = negative,
        noise_distribution = noise_distribution,
        w2v_subsampling = 0.0001,
        d2v_subsampling = 0,
        n2v_subsampling = 0,
        n2v_p = p,
        l2 = l2,
        batch_size = int(1e6),
        shuffle = shuffle,
        no_self_predict=0,
        tfidf = 0,
        undirected=undirected,
    )

    if total_samples is None:
        total_samples = max(len(model.graph),len(model.words)) * model.w2v_window * 100 * iteration

        # deepwalk & node2vec iter the network 5 times
        if "line" in alg.lower() or "n2v" in alg.lower():
            total_samples *= 5

    model.train(
        workers = workers,
        report_delay = report_delay,
        batch_size = int(1e6),
        total_samples = total_samples,
    )
    if output:
        logging.info("save embeddings to %s" % output)
        if "line" in alg.lower() or "n2v" in alg.lower():
            with open(output,'w') as f:
                f.write("%s %s\n" % (model.paper_embeddings.shape[0],model.paper_embeddings.shape[1]))
                for pid in range(model.paper_embeddings.shape[0]):
                    f.write("%s " % model.id2paper[pid])
                    f.write(" ".join([str(x) for x in model.paper_embeddings[pid]]))
                    f.write("\n")
        else:
            with open(output,'w') as f:
                f.write("%s %s\n" % (model.word_embeddings.shape[0],model.word_embeddings.shape[1]))
                for pid in range(model.word_embeddings.shape[0]):
                    f.write("%s " % model.id2word[pid])
                    f.write(" ".join([str(x) for x in model.word_embeddings[pid]]))
                    f.write("\n")
    return model
