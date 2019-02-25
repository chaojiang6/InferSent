import numpy as np
import matplotlib.pyplot as plt


def save_result_to_path(pred, path):
    print("A")
    with open(path, 'w') as f:
        for ele in pred:
            f.write(str(int(ele.cpu()))+"\n")

def URL_maxF1_eval(predict_result,test_data_label):
    test_data_label=[item>=1 for item in test_data_label]
    counter = 0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0
    
    for i, t in enumerate(predict_result):
        
        if t== True:
            guess=True
        else:
            guess=False
        label = test_data_label[i]
        #print guess, label
        if guess == True and label == False:
            fp += 1.0
        elif guess == False and label == True:
            fn += 1.0
        elif guess == True and label == True:
            tp += 1.0
        elif guess == False and label == False:
            tn += 1.0
        if label == guess:
            counter += 1.0
#else:
#print label+'--'*20
# if guess:
# print "GOLD-" + str(label) + "\t" + "SYS-" + str(guess) + "\t" + sent1 + "\t" + sent2

    try:
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        F = 2 * P * R / (P + R)
    except:
        P=0
        R=0
        F=0

    #print "PRECISION: %s, RECALL: %s, F1: %s" % (P, R, F)
    #print "ACCURACY: %s" % (counter/len(predict_result))
    accuracy=counter/len(predict_result)

    return (np.round(100*P,decimals = 2),np.round(100*R,decimals = 2),np.round(100*F,decimals = 2),np.round(100*accuracy,decimals = 2))

def save_results_to_figure(figure_path, result_dict):
    plt.figure(figsize=(25,5))
    plt.subplots_adjust(wspace =0.1, hspace =0.2)

    plt.subplot(1,3,1)
    x_1_1 = range(len(result_dict['train_acc']))
    y_1 = result_dict['train_acc']
    y_2 = result_dict['dev_acc']
    y_3 = result_dict['test_acc']
    plt.plot(x_1_1, y_1, 'r--x', label = 'Training')
    plt.xticks(x_1_1, [str(int(i)) for i in x_1_1])
    plt.plot(x_1_1, y_2, 'g-.o', label = 'Developing')
    plt.plot(x_1_1, y_3, 'b-^', label = 'Testing')
    plt.legend() # 展示图例
    plt.xlabel('Epoch') # 给 x 轴添加标签
    plt.ylabel('Accuracy %') # 给 y 轴添加标签
    plt.title('Accuracy in different stages') # 添加图形标题


    plt.subplot(1,3,2)
    x_1_1 = range(len(result_dict['train_task1_f1']))
    y_1 = result_dict['train_task1_f1']
    y_2 = result_dict['dev_task1_f1']
    y_3 = result_dict['test_task1_f1']
    plt.plot(x_1_1, y_1, 'r--x', label = 'Training')
    plt.xticks(x_1_1, [str(int(i)) for i in x_1_1])
    plt.plot(x_1_1, y_2, 'g-.o', label = 'Developing')
    plt.plot(x_1_1, y_3, 'b-^', label = 'Testing')
    plt.legend() # 展示图例
    plt.xlabel('Epoch') # 给 x 轴添加标签
    plt.ylabel('Task 1 F1') # 给 y 轴添加标签
    plt.title('Task 1 F1 in different stages') # 添加图形标题

    plt.subplot(1,3,3)
    x_1_1 = range(len(result_dict['train_task2_f1']))
    y_1 = result_dict['train_task2_f1']
    y_2 = result_dict['dev_task2_f1']
    y_3 = result_dict['test_task2_f1']
    plt.plot(x_1_1, y_1, 'r--x', label = 'Training')
    plt.xticks(x_1_1, [str(int(i)) for i in x_1_1])
    plt.plot(x_1_1, y_2, 'g-.o', label = 'Developing')
    plt.plot(x_1_1, y_3, 'b-^', label = 'Testing')
    plt.legend() # 展示图例
    plt.xlabel('Epoch') # 给 x 轴添加标签
    plt.ylabel('Task 2 F1') # 给 y 轴添加标签
    plt.title('Task 2 F1 in different stages') # 添加图形标题

    plt.savefig(figure_path, dpi = 300)


    # plt.show()
