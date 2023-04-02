# Amann

Amann, short for Amharic neural net, is a collection of tools that can be used to digitize amharic content. The scops of this project aren't set on stone and are continiously evovling but as of right now, the main goal is to build a tool that can convert a handwritten amharic text to a digitized version. To see progress and tasks you can contribute to head to the [contribution](#Contributing) section.

Amann, uses a CNN, to model the common amharic characters using the dataset openly avialble by [@Fetulhak](https://github.com/Fetulhak). A huge shout out to [@Fetulhak](https://github.com/Fetulhak) for collecting and cleaning such a huge collection of data. Using this dataset, Amann, currently, has a maximum accuracy of 86% on the test data and I look forward to improving it to atleast 96%. This will require improving the data set, and the algorithm. For more details again head to the contribution section.

If you are intersted about contributing or just want to chat, you can contact on [linkedin](https://www.linkedin.com/in/hileamlak-mulugeta-yitayew-a8b43317a/).

If you want a quick review of where the project is you can also look at the test_results folder. There you can find images along the models predictions.

## Table of Contents

- [Installation](#Installation)
- [Usage](https://chat.openai.com/chat#usage)
- [Contributing](https://chat.openai.com/chat#contributing)
- [License](https://chat.openai.com/chat#license)

## Installation

This project hasn't yet realeased an offical release, but if you want to experment with the current version, you can clone this repository, go to the src directory, and install the required liberaries using

`python -m install -r requirements.txt`

After that, you an play with the jupyter notebook inside the src folder.

## Contributing

I would appreciate any kind of help. This project is at its earliest stage, and could use a hand. If you are intersted checkout some of the TODOS below.

Searching for multiple amharic

* [ ] Rasterization and generating more images from fonts
* [ ] Create a dataset of different fontsizes and different image sizes (to be trained with adaptive pooling)
* [ ] Clearning up current data set
* [ ] Create a confusion matrix and other accuracy tests
* [ ] Do a better train test split making sure all classes are represented in both
* [ ] fine tune until getting >95% accuracy
* [ ] Build a pipeline to convert images to text
* [ ] Publish as a python package
* [ ] Support for uncommon characters
* [ ] Support for customized training
* [ ] Building pipline for training model from user data
* [ ] RNN for next word predictions


If you think you can help with any of these. Ping me, I would love to talk to you. I will soon be publishing a contribution policy to make sure we utilize man power properly.

## License

This program is distirbtued under GNU GENERAL PUBLIC LICENSE. For more information check the LICENSE file.

## Authors

Hileamlak Yitayew
