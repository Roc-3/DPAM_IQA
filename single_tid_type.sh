#!/bin/bash

for i in {1..24}
do
    # 修改配置文件中的路径
    sed -i "s|tid2013_label\": \"data/tid2013/type/type_[0-9]*.txt\"|tid2013_label\": \"data/tid2013/type/type_$i.txt\"|" single_tid_type.py
    sed -i "s|output_tid_type[0-9]*|output_tid_type$i|" single_tid_type.py

    # 运行 Python 脚本
    python single_tid_type.py
done

# chmod +x single_tid_type.sh
# ./single_tid_type.sh