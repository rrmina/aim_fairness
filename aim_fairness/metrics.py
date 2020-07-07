import numpy as np
from texttable import Texttable

def report_group_metrics(A, Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    assert Y_hat.shape == A.shape, "Y_hat {} and A {} are expected to have the same shape".format(Y_hat.shape, A.shape)
    assert len(A.shape) == 1, "A {}, Y_hat {}, and Y {} are expected to be 1D vectors".format(A.shape, Y_hat.shape, Y.shape)

    # Divide subgroups of labels and predictions
    Y_A0 = Y[A==0]              # Labels of Group 0
    Y_A1 = Y[A==1]              # Labels of Group 1
    Y_hat_A0 = Y_hat[A==0]      # Predictions of Group 0
    Y_hat_A1 = Y_hat[A==1]      # Predictions of Group 1

    # Confusion Matrix
    print("=" * 70)
    print("{}{}".format(" " * 26, "Confusion Matrices"))
    print("=" * 70)
    Confusion_matrix(Y_hat_A0, Y_A0, title='A=0')
    print()
    Confusion_matrix(Y_hat_A0, Y_A0, percentage=True, title='A=0')
    print()
    Confusion_matrix(Y_hat_A1, Y_A1, title='A=1')
    print()
    Confusion_matrix(Y_hat_A1, Y_A1, percentage=True, title='A=1')

    # Debug Header
    if (debug):
        print("=" * 70)
        print("{}{}".format(" " * 30, "Debug Logs"))
        print("=" * 70)

    # Compute metrics
    base_rate = BaseRate_gap(A, Y, debug)
    dp_gap = DemographicParity_gap(A, Y_hat, debug)
    eo_gap = EqualizedOdds_gap(A, Y_hat, Y, debug)
    eopp_gap = EqualOpportunity_gap(A, Y_hat, Y, debug)
    acc_gap = Accuracy_gap(A, Y_hat, Y, debug)
    f1_gap = F1Score_gap(A, Y_hat, Y, debug)

    # Pretty Printing    
    row_format = "{:>30}{:>30.4f}"
    data = []
    data.append(["Base Rate", base_rate])
    data.append(["Demographic Parity", dp_gap])
    data.append(["Equalized Odds", eo_gap])
    data.append(["Equal Opportunity Y0", eopp_gap[0]])
    data.append(["Equal Opportunity Y1", eopp_gap[1]])
    data.append(["Accuracy Parity", acc_gap])
    data.append(["F1 Scores Gap", f1_gap])

    # Print
    headers = ["Fairness Definitions", "Gap Values"]
    print("=" * (len(headers) * 30 + 10))
    print("{:>30}{:>30}".format(*headers))
    print("=" * (len(headers) * 30 + 10))
    for row in data:
        print(row_format.format(*row))

def TP(Y_hat, Y):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    correct = (Y_hat == Y) * 1
    return np.sum(correct[Y==1])

def TN(Y_hat, Y):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    correct = (Y_hat == Y) * 1
    return np.sum(correct[Y==0])

def FP(Y_hat, Y):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    wrong = 1 - (Y_hat == Y) * 1
    return np.sum(wrong[Y==0])

def FN(Y_hat, Y):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    wrong = 1 - (Y_hat == Y) * 1
    return np.sum(wrong[Y==1])

# Print Confusion Matrix
def Confusion_matrix(Y_hat, Y, percentage=False, title=''):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    tp, fp, fn, tn = TP(Y_hat, Y), FP(Y_hat, Y), FN(Y_hat, Y), TN(Y_hat, Y)

    if (percentage):
        sum_all = tp + fp + fn + tn
        tp, fp, fn, tn = tp / sum_all, fp / sum_all, fn / sum_all, tn / sum_all

    # String Format
    t = Texttable()
    t.add_rows([
        [    title,'Y=1', 'Y=0'],
        ['Y_hat=1',   tp,    fp],
        ['Y_hat=0',   fn,    tn]
    ])
    print(t.draw())

# Sensitivty, Recall, Hit Rate or True Positive Rate (TPR)
def Recall(Y_hat, Y):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    return np.mean(Y_hat[Y==1])

# Specificity, Selectivity, or True Negative Rate (TNR)
def Specificity(Y_hat, Y):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    return 1 - np.mean(Y_hat[Y==0])

# Precision or Positive Prediction Value
def Precision(Y_hat, Y):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    return np.mean(Y[Y_hat==1])

####################################################################################
#                                   Gap Scores
####################################################################################
def F1Score_gap(A, Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    assert Y_hat.shape == A.shape, "Y_hat {} and A {} are expected to have the same shape".format(Y_hat.shape, A.shape)
    assert len(A.shape) == 1, "A {}, Y_hat {}, and Y {} are expected to be 1D vectors".format(A.shape, Y_hat.shape, Y.shape)

    return FBetaScore_gap(A, Y_hat, Y, beta=1, debug=debug)

def FBetaScore_gap(A, Y_hat, Y, beta=1, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    assert Y_hat.shape == A.shape, "Y_hat {} and A {} are expected to have the same shape".format(Y_hat.shape, A.shape)
    assert len(A.shape) == 1, "A {}, Y_hat {}, and Y {} are expected to be 1D vectors".format(A.shape, Y_hat.shape, Y.shape)

    # Divide subgroups of labels
    Y_A0 = Y[A==0]              # Labels of Group 0
    Y_A1 = Y[A==1]              # Labels of Group 1
    Y_hat_A0 = Y_hat[A==0]      # Predictions of Group 0
    Y_hat_A1 = Y_hat[A==1]      # Predictions of Group 1

    if (debug):
        print("F{} Scores".format(beta))

    # Compute F1 Score for A0
    F_beta_score_A0 = FBeta_score(Y_hat_A0, Y_A0, debug=debug)
    if (debug):
        print("{:>5} F{} Score (A=0): {:>20.4f}".format("", beta, F_beta_score_A0))

    # Compute F1 Score for A1
    F_beta_score_A1 = FBeta_score(Y_hat_A1, Y_A1, debug=debug)
    if (debug):
        print("{:>5} F{} Score (A=1): {:>20.4f}".format("", beta, F_beta_score_A1))

    return abs(F_beta_score_A0 - F_beta_score_A1)

def F1_score(Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    return FBeta_score(Y_hat, Y, beta=1, debug=debug)

def FBeta_score(Y_hat, Y, beta=1, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    precision_ = Precision(Y_hat, Y)
    recall_ = Recall(Y_hat, Y)

    if (debug):
        print("{:>5} Precision: {:>20.4f}".format("", precision_))
        print("{:>5} Recall: {:>20.4f}".format("", recall_))

    constant = (1+ np.power(beta, 2))
    numerator = precision_ * recall_
    denominator = (np.power(beta, 2) * precision_) + recall_

    if (numerator == 0):
        return 0

    return constant * numerator / denominator

# DP_gap and BaseRate_gap are essentially the same, except for their inputs
def BaseRate_gap(A, Y, debug=False):
    assert Y.shape == A.shape, "Y {} and A {} are expected to have the same shape".format(Y.shape, A.shape)
    assert len(A.shape) == 1, "Y {} and A {} are expected to be 1D vectors".format(Y.shape, A.shape)

    # Divide subgroups of labels
    Y_A0 = Y[A==0]              # Labels of Group 0
    Y_A1 = Y[A==1]              # Labels of Group 1

    # Get the positive rates/Base Rates
    BaseRate0 = np.mean((Y_A0 == 1) * 1)
    BaseRate1 = np.mean((Y_A1== 1) * 1)

    if (debug):
        print("Base Rate")
        print("{:>5} Base Rate (A=0): {:>20.4f}".format("", BaseRate0))
        print("{:>5} Base Rate (A=1): {:>20.4f}".format("", BaseRate1))

    return abs(BaseRate0 - BaseRate1) 

# DP_gap and BaseRate_gap are essentially the same, except for their inputs
def DemographicParity_gap(A, Y_hat, debug=False):
    assert Y_hat.shape == A.shape, "Y_hat {} and A {} are expected to have the same shape".format(Y_hat.shape, A.shape)
    assert len(A.shape) == 1, "Y_hat {} and A {} are expected to be 1D vectors".format(Y_hat.shape, A.shape)

    # Divide subgroups of labels
    Y_hat_A0 = Y_hat[A==0]      # Predictions of Group 0
    Y_hat_A1 = Y_hat[A==1]      # Predictions of Group 1

    # Get the positive rates
    PosRate0 = np.mean((Y_hat_A0 == 1) * 1)
    PosRate1 = np.mean((Y_hat_A1== 1) * 1)

    if (debug):
        print("Demographic Parity")
        print("{:>5} Positive Rate (A=0): {:>20.4f}".format("", PosRate0))
        print("{:>5} Positive Rate (A=1): {:>20.4f}".format("", PosRate1))

    return abs(PosRate0 - PosRate1) 

# Accuracy Gap
def Accuracy_gap(A, Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    assert Y_hat.shape == A.shape, "Y_hat {} and A {} are expected to have the same shape".format(Y_hat.shape, A.shape)
    assert len(A.shape) == 1, "A {}, Y_hat {}, and Y {} are expected to be 1D vectors".format(A.shape, Y_hat.shape, Y.shape)

    # Divide subgroups of labels and predictions
    Y_A0 = Y[A==0]              # Labels of Group 0
    Y_A1 = Y[A==1]              # Labels of Group 1
    Y_hat_A0 = Y_hat[A==0]      # Predictions of Group 0
    Y_hat_A1 = Y_hat[A==1]      # Predictions of Group 1

    # Get the accuracies for both groups
    Acc0 = np.mean((Y_A0 == Y_hat_A0)*1)
    Acc1 = np.mean((Y_A1 == Y_hat_A1)*1)
    
    if (debug):
        print("Accuracy Parity")
        print("{:>5} Accuracy (A=0): {:>20.4f}".format("", Acc0))
        print("{:>5} Accuracy (A=1): {:>20.4f}".format("", Acc1))

    return abs(Acc0 - Acc1)

# Equalized Odds Gap
def EqualizedOdds_gap(A, Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    assert Y_hat.shape == A.shape, "Y_hat {} and A {} are expected to have the same shape".format(Y_hat.shape, A.shape)
    assert len(A.shape) == 1, "A {}, Y_hat {}, and Y {} are expected to be 1D vectors".format(A.shape, Y_hat.shape, Y.shape)

    # Divide subgroups of labels and predictions
    Y_A0 = Y[A==0]              # Labels of Group 0
    Y_A1 = Y[A==1]              # Labels of Group 1
    Y_hat_A0 = Y_hat[A==0]      # Predictions of Group 0
    Y_hat_A1 = Y_hat[A==1]      # Predictions of Group 1

    # Recall and Specificity
    TPR_A0 = Recall(Y_hat_A0, Y_A0)             # TPR = Recall
    TPR_A1 = Recall(Y_hat_A1, Y_A1)             
    FPR_A0 = 1 - Specificity(Y_hat_A0, Y_A0)    # FPR
    FPR_A1 = 1 - Specificity(Y_hat_A1, Y_A1)

    # Equal Opportunity on both sides
    Eopp_Y1 = abs(TPR_A0 - TPR_A1)              # Equal in terms of Recall/TPR
    Eopp_Y0 = abs(FPR_A0 - FPR_A1)              # Equal in terms of FPR

    if (debug):
        print("Equalized Odds")
        print("{:>5} Equal Opporunity (Y=1) - TPR/Recall".format(""))
        print("{:>5} {:>5} True Positive Rate (A=0): {:>20.4f}".format("", "", TPR_A0))
        print("{:>5} {:>5} True Positive Rate (A=1): {:>20.4f}".format("", "", TPR_A1))
        print("{:>5} Equal Opporunity (Y=0) - FPR".format(""))
        print("{:>5} {:>5} False Postive Rate (A=0): {:>20.4f}".format("", "", FPR_A0))
        print("{:>5} {:>5} False Postive Rate (A=1): {:>20.4f}".format("", "", FPR_A0))
    return Eopp_Y0 + Eopp_Y1

# Equal Opportunity Gap
def EqualOpportunity_gap(A, Y_hat, Y, debug=False):
    assert Y_hat.shape == Y.shape, "Y_hat {} and Y {} are expected to have the same shape".format(Y_hat.shape, Y.shape)
    assert Y_hat.shape == A.shape, "Y_hat {} and A {} are expected to have the same shape".format(Y_hat.shape, A.shape)
    assert len(A.shape) == 1, "A {}, Y_hat {}, and Y {} are expected to be 1D vectors".format(A.shape, Y_hat.shape, Y.shape)

    # Divide subgroups of labels and predictions
    Y_A0 = Y[A==0]              # Labels of Group 0
    Y_A1 = Y[A==1]              # Labels of Group 1
    Y_hat_A0 = Y_hat[A==0]      # Predictions of Group 0
    Y_hat_A1 = Y_hat[A==1]      # Predictions of Group 1

    # Recall and Specificity
    TPR_A0 = Recall(Y_hat_A0, Y_A0)             # TPR = Recall
    TPR_A1 = Recall(Y_hat_A1, Y_A1)             
    FPR_A0 = 1 - Specificity(Y_hat_A0, Y_A0)    # FPR
    FPR_A1 = 1 - Specificity(Y_hat_A1, Y_A1)

    # Equal Opportunity on both sides
    Eopp_Y1 = abs(TPR_A0 - TPR_A1)              # Equal in terms of Recall/TPR
    Eopp_Y0 = abs(FPR_A0 - FPR_A1)              # Equal in terms of FPR

    if (debug):
        print("Equalized Odds")
        print("{:>5} Equal Opporunity (Y=1) - TPR/Recall".format(""))
        print("{:>5} {:>5} True Positive Rate (A=0): {:>20.4f}".format("", "", TPR_A0))
        print("{:>5} {:>5} True Positive Rate (A=1): {:>20.4f}".format("", "", TPR_A1))
        print("{:>5} Equal Opporunity (Y=0) - FPR".format(""))
        print("{:>5} {:>5} False Postive Rate (A=0): {:>20.4f}".format("", "", FPR_A0))
        print("{:>5} {:>5} False Postive Rate (A=1): {:>20.4f}".format("", "", FPR_A0))

    return Eopp_Y0, Eopp_Y1