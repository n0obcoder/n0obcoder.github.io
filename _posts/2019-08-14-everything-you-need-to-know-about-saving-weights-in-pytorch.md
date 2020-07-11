---
title: "Everything You Need To Know About Saving Weights In PyTorch"
date: 2019-08-14
tags: [PyTorch, Python, Neural Networks, Artificial Intelligence, Data Science]
header:
  image: "/images/posts/2019-08-14-everything-you-need-to-know-about-saving-weights-in-pytorch/media_01_header.jpeg"
excerpt: "PyTorch, Python, Neural Networks, Artificial Intelligence, Data Science"
mathjax: "true"
---

What do we Deep Learning practitioners do once we are done with training our models ?

**We chill !!!**

*Hahhhaha* *Just Kidding…*

We either save the learnt weights or the entire model so that we could further train the model or maybe use the trained model for inference !

Next thing that you guys might be interested in knowing is, when do we save just the learnt weights and when do we save the whole model.

In this blog, we will try to find the answers to these questions.

I will keep it very straightforward and simple while explaining you the ins and outs of the art of saving a model’s architecture and it’s weights in PyTorch.

We will also learn how to access the different modules, *nn.Modules* to be precise, in any given PyTorch model .

So feel free to fork this kaggle kernel and play with the [**code** ](https://www.kaggle.com/n0obcoder/things-to-know-about-saving-weights-in-pytorch): )

Let’s get started !!!

We start off by importing the [*bare necessities](https://www.youtube.com/watch?v=5dhSdnDb3tk)* of coding using PyTorch.

<iframe src="https://medium.com/media/2b06f4ea6dc8d765e988947139cd2a39" frameborder=0></iframe>

Next, we define a CNN based model.

<iframe src="https://medium.com/media/0054cff18e6ac2c15f28027973559a8a" frameborder=0></iframe>

Let’s initialize and print *model* and see what’s inside it.

<iframe src="https://medium.com/media/8d64d32b3cb628ecdff45d4466e65f1f" frameborder=0></iframe>

Printing the *model* shows you it’s architecture. But we are going to dive deeper, for we are the Deep Learning practitioners!

We need to make sure that we understand what exactly lies inside our model.

There is a way to access each and every learnable Parameter of a model along with their names. By the way, a [*torch.nn.Parameter](https://pytorch.org/docs/stable/nn.html#torch.nn.Parameter)* is a Tensor subclass , which when used with [*torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)* gets automatically added to the list of its parameters and appears in e.g., in *parameters()* or *named_parameters()** ***iterator. Adding a *torch.nn.Tensor* on the other hand doesn’t have such an effect. More on that later !

Back to printing all the parameters of the model.

<iframe src="https://medium.com/media/3ca6cc0677bf2f7d452dfe1dc438b84b" frameborder=0></iframe>

Wooouhooouhooou ! So what did just happen here ?

Let’s get into the *named_parameters()* function.

model.named_parameters() itself is a generator. It returns the *name* and *param*, which are nothing but the name of the parameter and the parameter itself. Here, the returned param is *torch.nn.Parameter* class which is a kind of tensor. Since param is a type of tensor, it has *shape*** **and *requires_grad*** **attributes too. *param.shape* is simply the shape of the tensor and *param.requires_grad*** **is a boolean which tells if the parameter is learnable or not. Since all the params in the model have **requires_grad** **= True**, it means that all the parameters are learnable and will update on training the model. Had it been set to **False** for a any specific *param*, that parameter’s weight would not update on training the model.

So, *requires_grad* is the flag that you might want to change when you want to *train/freeze* a specific set of layers of your model.

Now we will try to freeze all but the last layer of the model. If we go through all the names of all the parameters of the model, we can see that the name of the last layer is *‘fc’* which stands for ‘fully connected’.

So let’s freeze all the parameters except the ones with their names *‘fc.weight’* or *‘fc.bias’*

<iframe src="https://medium.com/media/691f3137240cee571a13b0ed027ec865" frameborder=0></iframe>

We can verify that the desired changes have been made successfully by printing out the *requires_grad* for all the parameters of the model

<iframe src="https://medium.com/media/9b5adcb83f98b727a92c73c5b8cd6271" frameborder=0></iframe>

We can see that the desired changes have been made successfully !

So we have learnt how to change the *requires_grad* flag for any desired parameter of the model. And we have also learnt that doing so can come in very handy in situations where we want to *learn/freeze* the weights of some specific parameters/layers in a model.

We will now learn 2 of the widely known ways of saving a model’s weights/parameters.

1. torch.save(model.state_dict(), ‘weights_path_name.pth’) 
 It saves only** **the **weights** of the model

1. torch.save(model, ‘model_path_name.pth’) 
It saves the entire model (the **architecture as well as the weights**)

## What Is state_dict() And Where To Use It ?

We will first see how to write the syntax for *state_dict*. It’s pretty easy.

<iframe src="https://medium.com/media/a813ddbceec4fc3eff8b14368758ed97" frameborder=0></iframe>

Its just a python’s ordered dictionary.

But, printing this, would result in chaos. So we wouldn’t print the *state_dict* for the entire model here, but I encourage you guys to go ahead and print it out on your screens !

I guess it’s a good time to divert a little from the topic.

See, printing *help(model)* tells us that model is an instance of *nn.Module*

<iframe src="https://medium.com/media/b29606a9ac22789d94ad0882f527920a" frameborder=0></iframe>

It could also be verified by using python’s isinstance function

<iframe src="https://medium.com/media/28cb497a2fbda11008222f75dc418d97" frameborder=0></iframe>

Is *model.fc* also an instance of *nn.Module* ?

<iframe src="https://medium.com/media/1fe917162a5a38b4cf8d26ca11e5662c" frameborder=0></iframe>

Apparently yes !

But what exactly is *fc***, **and where does it come form ?

We can see what all *nn.Module* objects lie under the model

<iframe src="https://medium.com/media/0d90bd173e6c5f337cec2fcb7a9876c2" frameborder=0></iframe>

The *named_children()* applied on any *nn.Module* object returns all it’s immediate children (also *nn.Module* objects). Looking at the results of the above written piece of code, we know that *‘sequential’, ‘layer1’, ‘layer2’,* and *‘fc’* are all the children of model and all of these are *nn.Module* class objects. Now we all know where *‘fc’* is coming from.

And you know what ? *state_dict()* works on any *nn.Module* object and returns all it’s immediate children(of class *nn.Module*).

So let’s try the *state_dict()* function on the *‘fc’* layer of the model.

<iframe src="https://medium.com/media/0392daec174810f755506a5b0cc4122c" frameborder=0></iframe>

Remember that *model.fc.state_dict()* or any *nnModule.state_dict()* is an **ordered dictionary**. So iterating over it gives us the keys of the dictionary which can be used to access the parameter tensor which, by the way, is not a *nn.Module* object, but a simple *torch.Tensor* with a *shape* and *requires_grad* attribute.

So it must be noted that when we save the *state_dict()* of a *nn.Module* object e.g. the model, the *torch.Tensor* objects are saved !

This is how we save the *state_dict* of the entire model.

<iframe src="https://medium.com/media/47c9a09fba37dbde4d642089e05a3c5e" frameborder=0></iframe>

This makes a *‘weights_only.pth’* file in the working directory and it holds, in an ordered dictionary, the *torch.Tensor* objects of all the layers of the model.

We will try to load the saved weights now. But before we do that, we need to define the model architecture first. It makes sense to define the model first and then to load the weights in it because the saved information is **just** the weights and **not** the model architecture.

<iframe src="https://medium.com/media/87c522f9d08690269eef40ad75403886" frameborder=0></iframe>

Once the weights are loaded in the defined model, let’s check the *requires_grad* attribute of all the layers of model_new.

<iframe src="https://medium.com/media/05a747cf8ad24ad5f9329a47b9482ffd" frameborder=0></iframe>

Wait ! What ?

What happened to all the *requires_grad* flags that we had set for all the different layers ? It seems like all the *requires_grad* flags have been reset to **True**.

Actually, we never saved the *required_grad* flag of the parameters in the first place. Remember, a *state_dict*** **is simply a python dictionary object that maps each layer to its parameter tensor. It does not save the *requires_grad* attribute of the parameters.

So we would need to again make the necessary changes to the *requires_grad* attribute of all the parameters before resume training of the model for more epochs

## How To Save The Entire Model And When To Do It ?

Yes we have this second way of saving things, in which we can save the entire model too. By entire model, I mean the **architecture** of the model as well as it’s **weights**.

So we will resume from the point where we had frozen all but the last layer (the *‘fc’* layer) of the model and save the entire model.

<iframe src="https://medium.com/media/b8eb06ecc2e957ab971b57ddab09b925" frameborder=0></iframe>

This makes a *‘entire_model.pth’* file in the working directory and it contains the model architecture **as well as** the saved weights.

We will try to load the saved model now. And this time, we do not need to define the model architecture as the information about the model architecture is already stored in the saved file.

<iframe src="https://medium.com/media/63be818266fd50015de780adc7968e17" frameborder=0></iframe>

Once the model is loaded, let’s check the *requires_grad* attribute of all the layers of model_new.

<iframe src="https://medium.com/media/c0a22484ed073418e483ac46b7c34f0d" frameborder=0></iframe>

That is exactly what we wanted to see, isn’t it ? :D

So when we saved the entire model, we saved the *nn.Module* object and doing so saves the *requires_grad* flags of all it’s parameters too.

### I would strongly suggest you guys to fork this [**public kaggle kernel ](https://www.kaggle.com/n0obcoder/things-to-know-about-saving-weights-in-pytorch)**and play with the code, to get the feel of it !

## **Summary**

We learnt a lot of things in this blog.

1. Applying ***named_parameters()*** on an *nn.Module* object e.g. *model* or 
*model.layer2* or *model.fc* returns all the names and the respective parameters. These parameters are *nn.Parameter* (subclass of *torch.Tensor*) objects and therefore they have *shape* and *requires_grad* attributes.

<iframe src="https://medium.com/media/7a4b1f7ca69e4047f40f726d79353fdd" frameborder=0></iframe>

2. The ***requires_grad*** attribute of a *nn.Parameter* object (learnable parameter object) decides whether to train or freeze a particular parameter. For example, if we want to freeze the *layer1* of the model, we would use the following code.

<iframe src="https://medium.com/media/a6489116e1a35dabe15ef18eed67ccc6" frameborder=0></iframe>

3. Applying ***named_children()*** on any *nn.Module* object returns all it’s immediate children (also *nn.Module* objects).

<iframe src="https://medium.com/media/88afcdc68d7d94b02dd1fa1f312d9a91" frameborder=0></iframe>

4. A ***state_dict()*** of any *nn.Module* object e.g. *model* or *model.layer2* or *model.fc* is simply a python **ordered dictionary** object that maps each parameter to its parameter tensor (*torch.Tensor* object). The **keys** of this ordered dictionary are the names of the parameters, which can be used to access the respective parameter tensors.

<iframe src="https://medium.com/media/6f3aee709782b0dcf632435db16747ff" frameborder=0></iframe>

5. Saving a *nn.Module* object’s *state_dict* only **saves the** **weights** of the various parameters of that object and **not the model architecture**. Neither does it involve the **requires_grad** attribute of the weights. So before loading the *state_dict*, one must define the model first.

<iframe src="https://medium.com/media/e19837404cf182ce6cfac8220b003224" frameborder=0></iframe>

6. Entire model (*nn.Module* object) can also be saved which would include the **model architecture as well as its weights**. Since we are saving the *nn.Module* object, the **requires_grad** attribute is also **saved** this way. Also we don’t need to define the model architecture before loading the saved file since the saved file already has the model architecture saved in it.

<iframe src="https://medium.com/media/83d5b2ed9dea3d09a3a9d38eb3d02d98" frameborder=0></iframe>

7. Saving the *state_dict* can be used to only save the weights of the model. It doesn’t save the *required_grad* flag, whereas saving the entire model does save the model architecture, it’s weights and the *requires_grad* attributes of all its parameters.

8. Both *state_dict* as well as the entire model can be saved to make inferences.

I am writing this blog because I have learnt a lot by reading other’s blogs and I feel that I should also write and share my learnings and knowledge, as much as I can. So please leave your feedback in the comments section down below. Also I am new to writing blogs, so any suggestions on how to improve my writing would be appreciated ! :D
