from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import dataset


def tree_impor(in_path):
    data=dataset.read_csv(in_path)
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(data.X, data.y)
    feat_imp=clf.feature_importances_
    feat_imp/=sum(feat_imp)
    print(feat_imp)

tree_impor("../uci/vehicle")  
