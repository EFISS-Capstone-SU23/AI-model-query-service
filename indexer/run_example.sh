#!/bin/bash
cat <<EOF > option.json
{
    "new_index_database_version": "1.2.0",
    "model_path": "/media/thaiminhpv/Storage/MinhFileServer/Public-Filebrowser/RelaHash_weights/torchscripts/relahash_tf_efficientnetv2_b3_relahash_64_deepfashion2_200_0.0005_adam.pt"
}
EOF

cat <<EOF > database_info.txt
/home/thaiminhpv/Workspace/Code/Bodge/data/Screenshot from 2022-11-13 19-51-01.png
/home/thaiminhpv/Workspace/Code/Bodge/data/Screenshot from 2022-11-13 20-36-56.png
/home/thaiminhpv/Workspace/Code/Bodge/data/Screenshot from 2022-11-14 09-19-52.png
/home/thaiminhpv/Workspace/Code/Bodge/data/Screenshot from 2022-11-14 09-20-30.png
/home/thaiminhpv/Workspace/Code/Bodge/data/Screenshot from 2022-11-14 14-06-45.png
/home/thaiminhpv/Workspace/Code/Bodge/data/Screenshot from 2022-11-14 14-08-15.png
/home/thaiminhpv/Workspace/Code/Bodge/data/Screenshot from 2022-11-14 14-41-49.png
/home/thaiminhpv/Workspace/Code/Bodge/data/Screenshot from 2022-11-21 21-46-10.png
/home/thaiminhpv/Workspace/Code/Bodge/data/Screenshot from 2022-11-21 21-46-13.png
/home/thaiminhpv/Workspace/Code/Bodge/data/Screenshot from 2022-11-23 12-11-12.png
/home/thaiminhpv/Workspace/Code/Bodge/data/Screenshot from 2022-11-23 18-21-23.png
EOF

python main.py \
    --database database_info.txt \
    --dump_index_path index \
    --device cpu \
    --num_workers 4 \
    --batch_size 64 \
    --model_path /media/thaiminhpv/Storage/MinhFileServer/Public-Filebrowser/RelaHash_weights/torchscripts/relahash_tf_efficientnetv2_b3_relahash_64_deepfashion2_200_0.0005_adam.pt \
    --new_index_database_version 1.2.0

rm -v option.json
rm -v database_info.txt