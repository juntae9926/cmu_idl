# lstm (2048, 1024, 2, bidirectional=True)
python main.py --device "cuda:1" \
               --batch 32 \
               --lr 1e-3 \
               --save-model "checkpoint" \
               --epochs 10 \
               --addi True \
               --addi-model "checkpoint/val_9.03.pth"