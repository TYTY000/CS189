import threading
import numpy as np
import cv2 as cv

# 确定k值的函数
def select_k(sigma, percentage):
    total = sum(sigma)
    sumvar = 0
    k = 0
    while sumvar < total * percentage:
        sumvar += sigma[k]
        k += 1
    return k

# SVD函数
def SVD(image, p):
    # 对图像的每个颜色通道进行奇异值分解
    U_r, sigma_r, V_r = np.linalg.svd(image[:,:,0])
    U_g, sigma_g, V_g = np.linalg.svd(image[:,:,1])
    U_b, sigma_b, V_b = np.linalg.svd(image[:,:,2])
    # 选择前k个奇异值
    k_r = select_k(sigma_r, p)
    k_g = select_k(sigma_g, p)
    k_b = select_k(sigma_b, p)
    print("保留"+str(p*100)+"%信息，降维后的特征个数：", k_r, k_g, k_b, "\n")
    U_r_k = U_r[:, :k_r]
    sigma_r_k = np.diag(sigma_r[:k_r])
    V_r_k = V_r[:k_r, :]
    U_g_k = U_g[:, :k_g]
    sigma_g_k = np.diag(sigma_g[:k_g])
    V_g_k = V_g[:k_g, :]
    U_b_k = U_b[:, :k_b]
    sigma_b_k = np.diag(sigma_b[:k_b])
    V_b_k = V_b[:k_b, :]
    # 重构图像的每个颜色通道
    recon_image_r = np.dot(U_r_k, np.dot(sigma_r_k, V_r_k))
    recon_image_g = np.dot(U_g_k, np.dot(sigma_g_k, V_g_k))
    recon_image_b = np.dot(U_b_k, np.dot(sigma_b_k, V_b_k))
    # 将重构的颜色通道组合成一张图像
    recon_image = np.stack([recon_image_r, recon_image_g, recon_image_b], axis=2)
    return recon_image

def main():
    imagePath = '/mnt/d/屏幕截图 2023-08-18 191610.png'
    image = cv.imread(imagePath)
    print("降维前的特征个数："+str(image.shape[1])+"\n")
    print(image)
    print('----------------------------------------')
    reconImage = SVD(image, 0.99)
    reconImage = reconImage.astype(np.uint8)
    print(reconImage)
    cv.imshow('image', reconImage)
    while True:
        k = cv.waitKey(100)
        if cv.getWindowProperty('image',cv.WND_PROP_VISIBLE) < 1:
            break
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
# #数据中心化
# def Z_centered(dataMat):
# 	rows,cols=dataMat.shape
# 	meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
# 	meanVal = np.tile(meanVal,(rows,1))
# 	newdata = dataMat-meanVal
# 	return newdata, meanVal
#  
# #协方差矩阵
# def Cov(dataMat):
# 	meanVal = np.mean(data,0) #压缩行，返回1*cols矩阵，对各列求均值
# 	meanVal = np.tile(meanVal, (rows,1)) #返回rows行的均值矩阵
# 	Z = dataMat - meanVal
# 	Zcov = (1/(rows-1))*Z.T * Z
# 	return Zcov
# 	
# #最小化降维造成的损失，确定k
# def Percentage2n(eigVals, percentage):
# 	sortArray = np.sort(eigVals)  # 升序
# 	sortArray = sortArray[-1::-1]  # 逆转，即降序
# 	arraySum = sum(sortArray)
# 	tmpSum = 0
# 	num = 0
# 	for i in sortArray:
# 		tmpSum += i
# 		num += 1
# 		if tmpSum >= arraySum * percentage:
# 			return num
# 	
# #得到最大的k个特征值和特征向量
# def EigDV(covMat, p):
# 	D, V = np.linalg.eig(covMat) # 得到特征值和特征向量
# 	k = Percentage2n(D, p) # 确定k值
# 	print("保留"+str(p*100)+"%信息，降维后的特征个数：\n")
# 	eigenvalue = np.argsort(D)
# 	K_eigenValue = eigenvalue[-1:-(k+1):-1]
# 	K_eigenVector = V[:,K_eigenValue]
# 	return K_eigenValue, K_eigenVector
# 	
# #得到降维后的数据
# def getlowDataMat(DataMat, K_eigenVector):
# 	return DataMat * K_eigenVector
#  
# #重构数据
# def Reconstruction(lowDataMat, K_eigenVector, meanVal):
# 	reconDataMat = lowDataMat * K_eigenVector.T + meanVal
# 	return reconDataMat
#  
# #PCA算法
# def PCA(data, p):
# 	dataMat = np.float32(np.mat(data))
# 	#数据中心化
# 	dataMat, meanVal = Z_centered(dataMat)
# 	#计算协方差矩阵
# 		#covMat = Cov(dataMat)
# 	covMat = np.cov(dataMat, rowvar=0)
# 	#得到最大的k个特征值和特征向量
# 	D, V = EigDV(covMat, p)
# 	#得到降维后的数据
# 	lowDataMat = getlowDataMat(dataMat, V)
# 	#重构数据
# 	reconDataMat = Reconstruction(lowDataMat, V, meanVal)
# 	return reconDataMat

# # SVD函数
# def SVD(image, k):
#     # 对图像的每个颜色通道进行奇异值分解
#     U_r, sigma_r, V_r = np.linalg.svd(image[:,:,0])
#     U_g, sigma_g, V_g = np.linalg.svd(image[:,:,1])
#     U_b, sigma_b, V_b = np.linalg.svd(image[:,:,2])
#     # 选择前k个奇异值
#     U_r_k = U_r[:, :k]
#     sigma_r_k = np.diag(sigma_r[:k])
#     V_r_k = V_r[:k, :]
#     U_g_k = U_g[:, :k]
#     sigma_g_k = np.diag(sigma_g[:k])
#     V_g_k = V_g[:k, :]
#     U_b_k = U_b[:, :k]
#     sigma_b_k = np.diag(sigma_b[:k])
#     V_b_k = V_b[:k, :]
#     # 重构图像的每个颜色通道
#     recon_image_r = np.dot(U_r_k, np.dot(sigma_r_k, V_r_k))
#     recon_image_g = np.dot(U_g_k, np.dot(sigma_g_k, V_g_k))
#     recon_image_b = np.dot(U_b_k, np.dot(sigma_b_k, V_b_k))
#     # 将重构的颜色通道组合成一张图像
#     recon_image = np.stack([recon_image_r, recon_image_g, recon_image_b], axis=2)
#     return recon_image
# 
# def main():
#     imagePath = '/mnt/d/屏幕截图 2023-08-18 191610.png'
#     image = cv.imread(imagePath)
#     print("降维前的特征个数："+str(image.shape[1])+"\n")
#     print(image)
#     print('----------------------------------------')
#     reconImage = SVD(image, 50)  # 保留前50个奇异值
#     reconImage = reconImage.astype(np.uint8)
#     print(reconImage)
#     cv.imshow('image', reconImage)
#     while True:
#         k = cv.waitKey(100)
#         if cv.getWindowProperty('image',cv.WND_PROP_VISIBLE) < 1:
#             break
#     cv.destroyAllWindows()
# 
# if __name__ == "__main__":
#     main()
# image_lock = threading.Lock()

# def display_image(image):
#     with image_lock:
#         cv.imshow('image', image)
#         cv.waitKey(25)
#         cv.destroyAllWindows()
# def main():
#     imagePath = '/mnt/d/屏幕截图 2023-08-18 191610.png'
#     image = cv.imread(imagePath)
#     image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     rows,cols=image.shape
#     print("降维前的特征个数："+str(cols)+"\n")
#     print(image)
#     print('----------------------------------------')
#     reconImage = PCA(image, 0.99)
#     reconImage = reconImage.astype(np.uint8)
#     print(reconImage)
# # plt.imshow(reconImage, cmap='gray')
# # plt.show()
#     cv.imshow('image', reconImage)
#     while True:
#         k = cv.waitKey(100) # change the value from the original 0 (wait forever) to something appropriate
#         # if k == 27:
#         #     print('ESC')
#         #     cv.destroyAllWindows()
#         #     break
#         if cv.getWindowProperty('image',cv.WND_PROP_VISIBLE) < 1:
#             break
#     cv.destroyAllWindows()

# if __name__ == "__main__":
#     main()

