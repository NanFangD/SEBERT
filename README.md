# SEBERT
基于语义信息的命名实体识别模型，通过将语义信息融入到字向量中增强模型对长实体的检测能力，并通过混合loss提高模型的学习能力。
实验结果表明，该文提出的模型不仅适用于疫情文本数据，对比基线模型在4个中文数据集上F1值都有较高的提升。
COVID数据集已经放在了dataset中

#### requirements
python==3.6</br>
torch==1.10.0</br>
transformers==3.1.0</br>
tensorboard==2.10.0</br>
pandas==1.1.5</br>

#### 初始学习率设置
![image](https://user-images.githubusercontent.com/48280188/209778676-8a9393a9-2a21-41bc-92b6-9101607c589a.png)
#### 模型整体结构图如下：
![image](https://user-images.githubusercontent.com/48280188/209778827-91f18302-590a-484d-8aea-3af58db71a06.png)


#### 实验结果

![image](https://user-images.githubusercontent.com/48280188/209778530-893afd76-b5d9-4052-a271-84d430397d91.png)
![image](https://user-images.githubusercontent.com/48280188/209778549-8e9899f2-c1e2-4205-ad0a-b17394d05d11.png)
![image](https://user-images.githubusercontent.com/48280188/209778560-64386e01-4356-47bb-b619-7c040345837c.png)
![image](https://user-images.githubusercontent.com/48280188/209778570-5b5a24bc-bcff-47b9-a2aa-0aa5c127896b.png)
![image](https://user-images.githubusercontent.com/48280188/209778585-99a1cbc3-cf6b-4c6f-97d7-dd8f595401ee.png)

