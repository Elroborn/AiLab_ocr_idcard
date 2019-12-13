docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/ocr_idcard:$1 -f docker/stream_demos/Dockerfile .
docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/ocr_idcard:$1
