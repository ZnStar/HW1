from utils import *
import argparse
from model import *

def test(path):
    w,b,mu,sigma,nHidden,nLabels = load_model(random_initial=False,path=path)
    Xtest,ytest = load_data(train = False)
    Xtest, _, _ = standardizeCols(Xtest, mu, sigma)
    n = Xtest.shape[0]
    prob = MLPclassificationPredict(w, b, Xtest, nHidden, nLabels)
    yhat = prob.argmax(axis=1).reshape(prob.shape[0], 1)
    print('Test accuuracy with final model = {acc}'.format(
        acc=1-float(sum(yhat != ytest) / n)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--para_path', type=str, default='best_model.mat',
                        help='the path of best model parameters matrix')
    args = parser.parse_args()
    test(args.para_path)