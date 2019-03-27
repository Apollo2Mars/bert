
BERT_LARGE_DIR=/export/home/sunhongchao1/2-MRC/bert/chinese_L-12_H-768_A-12
SQUAD_DIR=/export/home/sunhongchao1/2-MRC/bert/data_squad

python3 run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=./output_squad \
  --use_tpu=False \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True
