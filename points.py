from collections import defaultdict
import matplotlib.pyplot as plt
import gauss,utils,visualize

class GaussExp(object):
    def __init__(self,feats=None):
        self.feats=None

    def __call__(self,in_path,out_path):
        utils.make_dir(out_path)
        @utils.DirFun({'in_path':0,'out_path':1})
        def helper(in_path,out_path):
            print(in_path)
            norm_cri,k=gauss.good_of_fit(in_path,
                                         alg_type="bayes",
                                         show=False)
            visualize.bar_plot(norm_cri,f'{out_path}_crit')
#            plt.savefig(f'{out_path}_crit')
            gauss.point_distribution(in_path,k=k,show=out_path)
#            plt.savefig(out_path)
        helper(in_path,out_path)

def stability_test(in_path,n_iters=100):
    size_counter=defaultdict(lambda:0)
    for i in range(n_iters):
        print(i)
        _,k=gauss.good_of_fit(in_path,
                              alg_type="bayes",
                              show=False)
        size_counter[k]=size_counter[k]+1
    print(size_counter)

GaussExp()("uci","gauss")