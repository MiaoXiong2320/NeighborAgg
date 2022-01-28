
# for RANDOMSEED in 1 2 3 4 5
# do
# python ./find_neighbors_classwise.py \
#     --src_folder "../classifier/code/simple_cls/output/fashionmnist/fashionmnist_smallerconvnet/feats_copy/e20/" \
#     --K 10
# done

for RANDOMSEED in 1 2 3 4 5
do
python ./show_neighbors_classwise.py \
    --dist_measure "Dot" \
    --src_folder "../classifier/code/simple_cls/output/fashionmnist/fashionmnist_smallerconvnet/feats_copy/e20/" \
    --K 10
done

for RANDOMSEED in 1 2 3 4 5
do
python ./show_neighbors_classwise.py \
    --dist_measure "L2" \
    --src_folder "../classifier/code/simple_cls/output/fashionmnist/fashionmnist_smallerconvnet/feats_copy/e20/" \
    --K 10
done