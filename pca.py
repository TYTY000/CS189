import torch
features = 100
samples = 200
p = 0.95
data = torch.randn(100, features) # Generate 100 random samples with 10 features each

# 确定k值的函数
def select_k(s, percentage):
    total = sum(s)
    sumvar = 0
    k = 0
    while sumvar < total * percentage:
        sumvar += s[k]
        k += 1
    return k

# SVD函数
def SVD(data, p):
    # 对数据进行奇异值分解
    U, S, V = torch.svd(data)
    # 选择前k个奇异值
    k = select_k(S, p)
    print("保留"+str(p*100)+"%信息，截取特征个数："+str(k))
    U_k = U[:, :k]
    S_k = torch.diag(S[:k])
    V_k = V[:, :k]
    # 重构数据
    recon_data = torch.mm(torch.mm(U_k, S_k), V_k.T)
    return recon_data, V_k

def main():
    data = torch.randn(samples, features) # Generate 100 random samples with 10 features each


    # Calculate the mean and standard deviation for standardization
    mean = torch.mean(data, dim=0)
    std_dev = torch.std(data, dim=0)

    # Perform standardization by subtracting the mean and dividing by standard deviation
    scaled_data = (data - mean) / std_dev

    # Perform Singular Value Decomposition
    recon_data, principal_components = SVD(scaled_data, p)  # 保留50%的数据

    # Printing the first 10 records of the original data
    print("Original Data (first 10 records):")
    print(data[:10])

    # Projecting the original data onto the reduced dimensional space
    projected_data = torch.mm(scaled_data, principal_components)
    print("\nProjected Data (First 10 Rows):")
    print(projected_data[:10])

if __name__ == "__main__":
    main()


# # Printing the first 10 records of the original data
# print("Original Data (first 10 records):")
# print(data[:features])

# # Calculate the mean and standard deviation for standardization
# mean = torch.mean(data, dim=0)
# std_dev = torch.std(data, dim=0)

# # Perform standardization by subtracting the mean and dividing by standard deviation
# scaled_data = (data - mean) / std_dev

# # Printing mean and standard deviation of the original data
# print("\nMean and Standard Deviation of Original Data:")
# print(f"Mean: \n{mean}")
# print(f"Standard Deviation: \n{std_dev}")

# # Printing the first 10 records of the standardized data
# print("\nStandardized Data (first 10 records):")
# print(scaled_data[:features])

# # Printing mean and standard deviation of the standardized data
# print("\nMean and Standard Deviation of Standardized Data:")
# print(f"Mean: \n{torch.mean(scaled_data, dim=0)}")
# print(f"Standard Deviation: \n{torch.std(scaled_data, dim=0)}")


# # Perform Singular Value Decomposition
# U, S, V = torch.svd(scaled_data)

# # Printing the components of the SVD with descriptions
# print("\nSingular Value Decomposition (SVD) Results:")

# print("\nMatrix U (first 10 rows): \nRepresents the orthogonal matrix in SVD, providing the basis vectors for the column space")
# print("of the original data.")
# print(U[:features])  # U is usually a tall matrix, so we are printing just the first few rows for brevity.

# print("\nVector S: \nContains the singular values of the original matrix, which are the square roots of the eigenvalues of the")
# print("matrix product (original data matrix T * original data matrix). These values represent the amount of variance")
# print("retained by each principal component.")
# print(S)

# print("\nMatrix V: \nContains the orthonormal basis for the row space of the original data matrix. These are essentially the ")
# print("principal components themselves.")
# print(V)


# k = select_k(S,p)
# principal_components = V[:, :k]
# print("\nSelected Principal Components:")
# print(principal_components)

# # Printing the first 10 records of the original data
# print("Original Data (first 10 records):")
# print(data[:features])

# # Projecting the original data onto the reduced dimensional space
# projected_data = torch.mm(scaled_data, principal_components)
# print("\nProjected Data (First 10 Rows):")
# print(projected_data[:features])
