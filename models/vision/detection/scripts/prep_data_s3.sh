S3_BUCKET=$1

############################################################
# Setup directories
############################################################

BASE_DIR=$HOME/SageMaker
mkdir -p $BASE_DIR/data/coco
mkdir -p $BASE_DIR/data/weights

############################################################
# Download all data files
############################################################

wget -O $BASE_DIR/data/coco/train2017.zip http://images.cocodataset.org/zips/train2017.zip
wget -O $BASE_DIR/data/coco/val2017.zip http://images.cocodataset.org/zips/val2017.zip
wget -O $BASE_DIR/data/coco/annotations_trainval2017.zip \
        http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -O $BASE_DIR/data/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
        https://github.com/keras-team/keras-applications/releases/download/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5

############################################################
# Decompress arrange and archive
############################################################

unzip -o -q $BASE_DIR/data/coco/train2017.zip -d $BASE_DIR/data/coco
unzip -o -q $BASE_DIR/data/coco/val2017.zip -d $BASE_DIR/data/coco
unzip -o -q $BASE_DIR/data/coco/annotations_trainval2017.zip -d $BASE_DIR/data/coco
rm $BASE_DIR/data/coco/*.zip
cd $BASE_DIR/data/
tar -cf coco.tar coco

############################################################
# Upload to S3
############################################################

aws s3 cp $BASE_DIR/data/coco.tar s3://${S3_BUCKET}/awsdet/data/coco/coco.tar
aws s3 cp $BASE_DIR/data/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    s3://${S3_BUCKET}/awsdet/data/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
