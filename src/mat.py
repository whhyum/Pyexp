# def get_input_xy(src=[]):
#     pre_x = []
#     true_y = []
#
#     class_indices = {'cat': 0, 'dog': 1}
#
#     for s in src:
#         input = cv2.imread(s)
#         input = cv2.resize(input, (150, 150))
#         input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
#         pre_x.append(input)
#
#         _, fn = os.path.split(s)
#         y = class_indices.get(fn[:3])
#         true_y.append(y)
#
#     pre_x = np.array(pre_x) / 255.0
#
#     return pre_x, true_y
#
#
# def plot_sonfusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#
#     thresh = cm.max() / 2.0
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
#
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predict label')
#
# test = os.listdir(test_dir)
#
# images = []
#
# # 获取每张图片的地址，并保存在列表images中
# for testpath in test:
#     for fn in os.listdir(os.path.join(test_dir, testpath)):
#         if fn.endswith('jpg'):
#             fd = os.path.join(test_dir, testpath, fn)
#             images.append(fd)
#
# # 得到规范化图片及true label
# pre_x, true_y = get_input_xy(images)
#
# # 预测
# pred_y = model.predict_classes(pre_x)
#
# # 画混淆矩阵
# confusion_mat = confusion_matrix(true_y, pred_y)
# plot_sonfusion_matrix(confusion_mat, classes=range(2))