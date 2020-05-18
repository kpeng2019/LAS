import warnings
import shap
warnings.filterwarnings('ignore')

def SHAPplot(clf,X_train,X_test):
    shap.initjs()
    X_train_summary = shap.kmeans(X_train, 10)
    rf_explainer = shap.KernelExplainer(clf.predict, X_train_summary)
    shap_values_RF_test = rf_explainer.shap_values(X_test)
    # shap.force_plot(rf_explainer.expected_value, shap_values_RF_test, X_test)
    shap.summary_plot(shap_values_RF_test, X_test)
    return