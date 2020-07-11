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

<script src="https://gist.github.com/n0obcoder/3a093926545de23d6ce78146ab40a23d.js"></script>

Next, we define a CNN based model.

<script src="https://gist.github.com/n0obcoder/3ba47013e21ce4628cb6f6220d30880f.js"></script>

Let’s initialize and print *model* and see what’s inside it.

<script src="https://gist.github.com/n0obcoder/c554a53027732aa5799a5aa1c4660144.js"></script>

Printing the *model* shows you it’s architecture. But we are going to dive deeper, for we are the Deep Learning practitioners!

We need to make sure that we understand what exactly lies inside our model.

There is a way to access each and every learnable Parameter of a model along with their names. By the way, a [*torch.nn.Parameter](https://pytorch.org/docs/stable/nn.html#torch.nn.Parameter)* is a Tensor subclass , which when used with [*torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)* gets automatically added to the list of its parameters and appears in e.g., in *parameters()* or *named_parameters()** ***iterator. Adding a *torch.nn.Tensor* on the other hand doesn’t have such an effect. More on that later !

Back to printing all the parameters of the model.

<script src="https://gist.github.com/n0obcoder/e988501a9e3b0a234d07f0f71abd6956.js"></script>

Wooouhooouhooou ! So what did just happen here ?

Let’s get into the *named_parameters()* function.

model.named_parameters() itself is a generator. It returns the *name* and *param*, which are nothing but the name of the parameter and the parameter itself. Here, the returned param is *torch.nn.Parameter* class which is a kind of tensor. Since param is a type of tensor, it has *shape*** **and *requires_grad*** **attributes too. *param.shape* is simply the shape of the tensor and *param.requires_grad*** **is a boolean which tells if the parameter is learnable or not. Since all the params in the model have **requires_grad** **= True**, it means that all the parameters are learnable and will update on training the model. Had it been set to **False** for a any specific *param*, that parameter’s weight would not update on training the model.

So, *requires_grad* is the flag that you might want to change when you want to *train/freeze* a specific set of layers of your model.

Now we will try to freeze all but the last layer of the model. If we go through all the names of all the parameters of the model, we can see that the name of the last layer is *‘fc’* which stands for ‘fully connected’.

So let’s freeze all the parameters except the ones with their names *‘fc.weight’* or *‘fc.bias’*

<script src="https://gist.github.com/n0obcoder/9b70c73065734d4ee189ea15e82a3fbd.js"></script>

We can verify that the desired changes have been made successfully by printing out the *requires_grad* for all the parameters of the model

<script src="https://gist.github.com/n0obcoder/4e9a2a436f03cdc827788afb5da1a7c2.js"></script>

We can see that the desired changes have been made successfully !

So we have learnt how to change the *requires_grad* flag for any desired parameter of the model. And we have also learnt that doing so can come in very handy in situations where we want to *learn/freeze* the weights of some specific parameters/layers in a model.

We will now learn 2 of the widely known ways of saving a model’s weights/parameters.

1. torch.save(model.state_dict(), ‘weights_path_name.pth’) 
 It saves only** **the **weights** of the model

1. torch.save(model, ‘model_path_name.pth’) 
It saves the entire model (the **architecture as well as the weights**)

## What Is state_dict() And Where To Use It ?

We will first see how to write the syntax for *state_dict*. It’s pretty easy.

<script src="https://gist.github.com/n0obcoder/de97028a3fa48cd1aaf1c33d1a145750.js"></script>

Its just a python’s ordered dictionary.

But, printing this, would result in chaos. So we wouldn’t print the *state_dict* for the entire model here, but I encourage you guys to go ahead and print it out on your screens !

I guess it’s a good time to divert a little from the topic.

See, printing *help(model)* tells us that model is an instance of *nn.Module*

<script src="https://gist.github.com/n0obcoder/a8024ac9e796c9dffd3cc7386493ce5f.js"></script>

It could also be verified by using python’s isinstance function

<script src="https://gist.github.com/n0obcoder/c9adc94c50ae858643e1fabde1af39c3.js"></script>

Is *model.fc* also an instance of *nn.Module* ?

<script src="https://gist.github.com/n0obcoder/13a4f4398eb0737053778a7803c5dbfd.js"></script>

Apparently yes !

But what exactly is *fc***, **and where does it come form ?

We can see what all *nn.Module* objects lie under the model

<script src="https://gist.github.com/n0obcoder/4ffcdbbce2434574d88ef7cd875091f7.js"></script>

The *named_children()* applied on any *nn.Module* object returns all it’s immediate children (also *nn.Module* objects). Looking at the results of the above written piece of code, we know that *‘sequential’, ‘layer1’, ‘layer2’,* and *‘fc’* are all the children of model and all of these are *nn.Module* class objects. Now we all know where *‘fc’* is coming from.

And you know what ? *state_dict()* works on any *nn.Module* object and returns all it’s immediate children(of class *nn.Module*).

So let’s try the *state_dict()* function on the *‘fc’* layer of the model.

<script src="https://gist.github.com/n0obcoder/9baa58eed9aa6325ace326fe6bcf0b85.js"></script>

Remember that *model.fc.state_dict()* or any *nnModule.state_dict()* is an **ordered dictionary**. So iterating over it gives us the keys of the dictionary which can be used to access the parameter tensor which, by the way, is not a *nn.Module* object, but a simple *torch.Tensor* with a *shape* and *requires_grad* attribute.

So it must be noted that when we save the *state_dict()* of a *nn.Module* object e.g. the model, the *torch.Tensor* objects are saved !

This is how we save the *state_dict* of the entire model.

<script src="https://gist.github.com/n0obcoder/60d23fbf4994d093a8cc1e37b242cede.js"></script>

This makes a *‘weights_only.pth’* file in the working directory and it holds, in an ordered dictionary, the *torch.Tensor* objects of all the layers of the model.

We will try to load the saved weights now. But before we do that, we need to define the model architecture first. It makes sense to define the model first and then to load the weights in it because the saved information is **just** the weights and **not** the model architecture.

<script src="https://gist.github.com/n0obcoder/87d008fb5e7747fc380ffe2fa389b7d5.js"></script>

Once the weights are loaded in the defined model, let’s check the *requires_grad* attribute of all the layers of model_new.

<script src="https://gist.github.com/n0obcoder/ac9e3b97226669d139ef0614102f3c86.js"></script>

Wait ! What ?

What happened to all the *requires_grad* flags that we had set for all the different layers ? It seems like all the *requires_grad* flags have been reset to **True**.

Actually, we never saved the *required_grad* flag of the parameters in the first place. Remember, a *state_dict*** **is simply a python dictionary object that maps each layer to its parameter tensor. It does not save the *requires_grad* attribute of the parameters.

So we would need to again make the necessary changes to the *requires_grad* attribute of all the parameters before resume training of the model for more epochs

## How To Save The Entire Model And When To Do It ?

Yes we have this second way of saving things, in which we can save the entire model too. By entire model, I mean the **architecture** of the model as well as it’s **weights**.

So we will resume from the point where we had frozen all but the last layer (the *‘fc’* layer) of the model and save the entire model.

<script src="https://gist.github.com/n0obcoder/f585f87eb78a7db208b9218d7bc8e706.js"></script>

This makes a *‘entire_model.pth’* file in the working directory and it contains the model architecture **as well as** the saved weights.

We will try to load the saved model now. And this time, we do not need to define the model architecture as the information about the model architecture is already stored in the saved file.

<script src="https://gist.github.com/n0obcoder/f539676873bab027449367b559edb8a1.js"></script>

Once the model is loaded, let’s check the *requires_grad* attribute of all the layers of model_new.

<script src="https://gist.github.com/n0obcoder/31a23a8cd2db69148f8a938607684cd3.js"></script>

That is exactly what we wanted to see, isn’t it ? :D

So when we saved the entire model, we saved the *nn.Module* object and doing so saves the *requires_grad* flags of all it’s parameters too.

### I would strongly suggest you guys to fork this [**public kaggle kernel ](https://www.kaggle.com/n0obcoder/things-to-know-about-saving-weights-in-pytorch)**and play with the code, to get the feel of it !

## **Summary**

We learnt a lot of things in this blog.

1. Applying ***named_parameters()*** on an *nn.Module* object e.g. *model* or 
*model.layer2* or *model.fc* returns all the names and the respective parameters. These parameters are *nn.Parameter* (subclass of *torch.Tensor*) objects and therefore they have *shape* and *requires_grad* attributes.

<script src="https://gist.github.com/n0obcoder/ac7ad1c38d372389be88c32120997171.js"></script>

2. The ***requires_grad*** attribute of a *nn.Parameter* object (learnable parameter object) decides whether to train or freeze a particular parameter. For example, if we want to freeze the *layer1* of the model, we would use the following code.

<script src="https://gist.github.com/n0obcoder/07a47b4bffe723b7784e43209e8893af.js"></script>

3. Applying ***named_children()*** on any *nn.Module* object returns all it’s immediate children (also *nn.Module* objects).

<script src="https://gist.github.com/n0obcoder/37873ede4f6b129cc502dd55be4e4976.js"></script>

4. A ***state_dict()*** of any *nn.Module* object e.g. *model* or *model.layer2* or *model.fc* is simply a python **ordered dictionary** object that maps each parameter to its parameter tensor (*torch.Tensor* object). The **keys** of this ordered dictionary are the names of the parameters, which can be used to access the respective parameter tensors.

<script src="https://gist.github.com/n0obcoder/bc4f01784f657b53f84be8c654715ebf.js"></script>

5. Saving a *nn.Module* object’s *state_dict* only **saves the** **weights** of the various parameters of that object and **not the model architecture**. Neither does it involve the **requires_grad** attribute of the weights. So before loading the *state_dict*, one must define the model first.

<script src="https://gist.github.com/n0obcoder/1c2137b473e8d6623ca3660038db4ae9.js"></script>

6. Entire model (*nn.Module* object) can also be saved which would include the **model architecture as well as its weights**. Since we are saving the *nn.Module* object, the **requires_grad** attribute is also **saved** this way. Also we don’t need to define the model architecture before loading the saved file since the saved file already has the model architecture saved in it.

<script src="https://gist.github.com/n0obcoder/e8677e884d282c45754e028f6676187e.js"></script>

7. Saving the *state_dict* can be used to only save the weights of the model. It doesn’t save the *required_grad* flag, whereas saving the entire model does save the model architecture, it’s weights and the *requires_grad* attributes of all its parameters.

8. Both *state_dict* as well as the entire model can be saved to make inferences.

I am writing this blog because I have learnt a lot by reading other’s blogs and I feel that I should also write and share my learnings and knowledge, as much as I can. So please leave your feedback in the comments section down below. Also I am new to writing blogs, so any suggestions on how to improve my writing would be appreciated ! :D
