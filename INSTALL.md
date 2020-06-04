

# Installing

## Using pip

```
sudo pip3 install LASExplanation
```

## How to test the installation

```
python
>>>import LASExplanation.LIMEBAG as lm
>>>lm.demo1()
>>>lm.demo2()
```

## How to interpret the demo result
### Demo1()
1. The function first prints out attributes of the data explained. 
2. The index of each attribute will be used to report feature importance rank of each instance as seen in the following output.
3. "Number of unfair instances: 0 out of all 31 instances" reports the number of unfairly treated instances found by our method.
4. Finally, the demo prints out the feature importance ranks as well as weights for all data records explained.
- For example: [13, 8, 6, 14, 18, 17, 9, 10, 16, 1, 2, 5, 3, 4, 12, 0, 11, 15, 7, 19] means the 13th feature is most important in the prediction of this instance; the 8th feature is the second most important feature and so on.




### Demo2()
- Demo2() repeats the first 3 steps of demo1().
- Then it visualized the feature importance ranks&weights using scott-knott, an effect size test. The medians and IQRs of the 2 values are shown in the text, as well as the ranks determined by scott-knott. Rank 1 marks the most impotance feature(s). 
- A latex-version table is also generated in case it's needed in research study.


