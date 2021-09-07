var=0
while [ $var -eq 0 ]
do
    gpu_id=0
    #检测每个gpu的显存占用大小
    for i in $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    do
        if [ $i -lt 2000 ] #如果占用的显存大小在2000 MB以下
        then
            echo 'GPU'$gpu_id' is available'
            #此处运执行实验的脚本
            sh run.sh
            var=1
            break
        fi
        gpu_id=$(($gpu_id+1))
    done
done
