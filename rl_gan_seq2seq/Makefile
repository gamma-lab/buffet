help:
	@echo "    train"
	@echo "        train the dialogue conversation."


# --------------------- Train ---------------------
train:
	python -u irl-gan.py -data_id dailydialog -vocab_size 0 -hidden_size 1024 -ent_weight 0.005 -exp_id 2 --adv_train --teacher_forcing --no_testing --no_continue_train
