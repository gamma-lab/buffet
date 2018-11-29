# Lyrics generation using RNN 
This tutorial is modified from https://github.com/SeanLee97/generate-lyrics-using-PyTorch

The original code is written in Pytorch 2, which has been made compatible with Pytorch 3.

## Data:

The dataset is consisting of 7387 lines of Chinese lines lyrics from Jay Chou, which has been removed non-Chinese letter and symbols.


## Process Understanding:

### Build word dictionary
* Built a character to index and index to charactor dictionaries, in this way, we can get all the unique words from underlying data and assign each word with a unique index. Thus, we can transfer all the words into indexes for the training purpose and then transform back for the visualizing the result.
```
想要有直升机
想要和你飞到宇宙去
想要和你融化在一起
融化在银河里
我每天每天每天在想想想想着你
这样的甜蜜
让我开始相信命运
感谢地心引力
让我碰到你
漂亮的让我面红的可爱女人
```
### Training data generation
* To generate training data for each epoch, we build a random chunk of size 200 for each epoch and then transform the underlying Chinese data to indexes and then to tensor format for the prepartion for the input to RNN model
### Model Implementation
* The underlying model is a one-layer GRU model, with hidden size 128, learning rate 0.001, and dropout rate 0.1, you can always turn these parameters in order to generate a better performance.
### Training and Evaluation Process
* During the training process, the model is fed one word each time then the loss is calculated using the CrossEntropy loss by comparing the output word and the word after the input word, at the end, the loss is averaged for each chunk and printed out. for every 100 epochs. The model trained is saved during the training time and the training process can be terminated and restart.
* During the evaluation process, you can specify the words you want to start with and the length of words you want to generate. 

## Result Example:
```
>length 100
>input 听话
>output 听话 你会听爸爸话 都不再亲吻你怎么要好吗 我说的爱情请你要我的窝 我很后悔是有你的脸 说 一定都有你的烦恼 你说你是最美了 要我学会预治你大家的好像不断不能流力气 在我的怀里 再走一半在我的胸口 我们的 
>input 给我一首歌
>output 给我一首歌的时间 紧紧的把那拥抱更多一起逛 沉默的脸孔 你专辑该真爱要 但是要打倒抽搐 友一朵听着一口 时光机 我会降下掉听的讲 想要超越自然 那鲁湾  哦跑道理就要看到 坐着爱上的时间的轨迹 在风中黄昏似着  
>input 牛仔
>output 牛仔很忙 女人写世界的纪录 差着你的脸 轻刷著和弦 初恋是整遍 手写的永远 都枯萎 凋谢 冲向海边  我轻轻地尝一口 你说的爱 要爱有你 越要怎么连 我微笑的很好笑 你都有 不要放 竟太多的自由 到底 我 

.....
```
## Future Improvement:
The training time for `10000` epochs using `GPU` is around `100 mins`, you can increase the number of epochs and turn different model parameters to increase the performance.

## Note:
The comments for each function and the dimention of each layer in the RNN model are specified in the notebook. 
