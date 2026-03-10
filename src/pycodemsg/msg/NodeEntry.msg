string name
string alias # 别名
bool is_running  # 是否启动 
bool is_inner    # 内置节点
string command   # 启动命令
string package   # 包名
CommEntry[] publishers    # 发布主题
CommEntry[] subscribers   # 订阅主题
CommEntry[] services      # 服务
CommEntry[] actions       # 行为