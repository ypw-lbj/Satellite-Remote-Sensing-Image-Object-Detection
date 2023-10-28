min_ratio = 20
max_ratio = 90
feature_map_num = 4
min_dim = 512
step = (max_ratio-min_ratio)//(feature_map_num-2)
min_sizes = []
max_sizes = []
for ratio in range(min_ratio,max_ratio+1,step):
	min_sizes.append(min_dim*ratio/100.)
	max_sizes.append(min_dim*(ratio+step)/100.)
min_sizes = [min_dim*10./100]+min_sizes
max_sizes = [min_dim*20./100]+max_sizes
print(min_sizes)
print(max_sizes)
