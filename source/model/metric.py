import torch


def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)



# def CornerPrecision(output,target):
#     pred = outputs['pred']
#     labels = inputs['keypoint_map']
#
#     precision = tf.reduce_sum(pred*labels) / tf.reduce_sum(pred)
#     recall = tf.reduce_sum(pred*labels) / tf.reduce_sum(labels)
#
#     return {'precision': precision, 'recall': recall}
