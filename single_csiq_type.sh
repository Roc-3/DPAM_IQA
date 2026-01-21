#!/bin/bash

for i in {1..6}
do
    # 修改配置文件中的路径
    sed -i "s|csiq_label\": \"data/CSIQ/type/type_[0-9]*.txt\"|csiq_label\": \"data/CSIQ/type/type_$i.txt\"|" single_csiq_type.py
    sed -i "s|output_csiq_type[0-9]*|output_csiq_type$i|" single_csiq_type.py

    # 运行 Python 脚本
    python single_csiq_type.py
done

# chmod +x single_csiq_type.sh
# ./single_csiq_type.sh