import math

# 手动输入
peak =0.055941
target = 0.054359877  # 比较基准值/稳态值

# 计算超调量
Mp = (peak - target) / target

if Mp <= 0:
    print(f"阻尼比 zeta = 1.000000 (无超调)")
else:
    zeta = -math.log(Mp) / math.sqrt(math.pi**2 + math.log(Mp)**2)
    print(f"超调量 Mp = {Mp:.6f}")
    print(f"阻尼比 zeta = {zeta:.6f}")
