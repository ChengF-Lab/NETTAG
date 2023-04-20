
PPI="ppi_remove_self_loop.txt"
FOLDER="Human_350K"
# SEED=0
DEVICE=0
K=1200

echo "PPI    = $PPI"
echo "FOLDER = $FOLDER"
# echo "SEED   = $SEED"
echo "DEVICE = $DEVICE"
echo "K      = $K"


for SEED in {1..10}
do
  echo "SEED = $SEED"
  python ./main.py \
    --rand_seed $SEED \
    --device $DEVICE \
    --pretrained 0 \
    --dirnet      ../data/$PPI \
    --dirresult   ../output/$FOLDER/$SEED/ \
    --dirlog      ../output/$FOLDER/$SEED/ \
    --dirfuncgeno ../data/functional_genomics.conf \
    --preprocess svd \
    --K $K \
    --n_comp 3800 \
    --lr 1e-4 \
    --dropout 0.0 \
    --hidden_size 2048 1024
done

python ./ensemble.py ../output/$FOLDER/ 200
