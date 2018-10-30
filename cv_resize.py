import cv2
import os

## resize images
# pic_list = os.listdir('data/VOC2012/train')
#
# for i in range(len(pic_list)):
#
#     image = cv2.imread('D:/pythonWorkplace/SRGAN-master/data/VOC2012/big/train/'+pic_list[i])
#     image_resize = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
#     # cv2.imshow('image1',image)
#     # cv2.imshow('image2',image_resize)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     cv2.imwrite('D:/pythonWorkplace/SRGAN-master/data/VOC2012/mid/train/00000'+str(i+1)+'.jpg',
#                 image_resize
#                 )


## rename files
# import os
# path = 'data/VOC2012/train'
# count = 1
# for file in os.listdir(path):
#     os.rename(os.path.join(path,file),os.path.join(path,  str(count)+".jpg"))
#
#     count+=1


# seek image
pic_list = os.listdir('data/VOC2012/train')
pic_list = sorted(pic_list)

for i in range(0, len(pic_list)):

    image = cv2.imread('data/VOC2012/train/' + pic_list[i])

    if image.shape[0] <=88 or image.shape[1] <= 88:
        print("filename %s" % pic_list[i])
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # if image.shape[0] > image.shape[1] and image.shape[0] < 500:
    #     print("num: %d, height: %d, filename %s" % (i, image.shape[0], pic_list[i]))
    #     cv2.imshow('image', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # elif image.shape[0] < image.shape[1] and image.shape[1] < 500:
    #     print("num: %d, height: %d, filename %s" % (i, image.shape[0], pic_list[i]))
    #     cv2.imshow('image', image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()



print("OK")