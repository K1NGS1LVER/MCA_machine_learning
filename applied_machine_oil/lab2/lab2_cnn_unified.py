# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Lab 2 — Unified CNN Exploration
#
# **What:** One reproducible notebook connecting grayscale medical images, RGB natural images, learned CNN features, and LeNet-5.
#
# **Why:** The same convolution ideas become clearer when we compare hand-crafted filters, random kernels, and trained features while tracking shape, complexity, and receptive field.

# %% [markdown]
# ## Conventions and execution notes
#
# - **What/Why:** Every section states what it does and why it matters.
# - **Shapes:** Images use `(batch, height, width, channels)`; latent vectors use `(batch, features)`.
# - **Feature vocabulary:** *hand-crafted* = fixed human-designed kernels; *random* = untrained initialized kernels; *learned* = weights updated from data.
# - **Data safety:** validation is split from training data; test data is held out for final evaluation.
# - **Optional resources:** TissueMNIST, KerasTuner, and ImageNet weights have explicit fallback messages instead of silent failure.
# - Run this notebook from this directory in the linked project venv so `sons_of_anarchy.jpg` resolves relatively.

# %%
import os, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
SEED=42
os.environ["PYTHONHASHSEED"]=str(SEED); random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
ROOT=Path.cwd(); IMAGE_PATH=ROOT/"sons_of_anarchy.jpg"
print(f"Python {sys.version.split()[0]} | TensorFlow {tf.__version__}"); print(f"Working directory: {ROOT}")
assert IMAGE_PATH.exists(), f"Expected relative image path: {IMAGE_PATH}"


# %% [markdown]
# ## Shared helpers
#
# **What:** Define one arbitrary odd-kernel correlation helper, a true-convolution wrapper, plotting helpers, and complexity/receptive-field calculations.
#
# **Why:** Keeping these primitives shared prevents duplicated logic and makes manual versus built-in operations explicit.

# %%
def apply_odd_kernel(image,kernel,padding="reflect"):
    image=np.asarray(image,dtype=np.float32); kernel=np.asarray(kernel,dtype=np.float32)
    assert image.ndim==2 and kernel.ndim==2 and kernel.shape[0]==kernel.shape[1] and kernel.shape[0]%2==1
    radius=kernel.shape[0]//2; padded=np.pad(image,radius,mode=padding)
    return np.array([[np.sum(padded[r:r+2*radius+1,c:c+2*radius+1]*kernel) for c in range(image.shape[1])] for r in range(image.shape[0])],dtype=np.float32)
def true_convolution(image,kernel,padding="reflect"):
    return apply_odd_kernel(image,np.flip(np.asarray(kernel),axis=(0,1)),padding)
def show_maps(images,titles,cmap="viridis",cols=4,figsize=None):
    images=list(images); rows=int(np.ceil(len(images)/cols)); fig,axes=plt.subplots(rows,cols,figsize=figsize or (3.2*cols,3*rows),squeeze=False)
    for ax,image,title in zip(axes.ravel(),images,titles): ax.imshow(image,cmap=cmap); ax.set_title(f"{title}\n{np.shape(image)}"); ax.axis("off")
    for ax in axes.ravel()[len(images):]: ax.axis("off")
    plt.tight_layout(); return fig
def cnn_complexity(model,input_shape):
    x=tf.zeros((1,*input_shape)); rows=[]; receptive=jump=1
    for layer in model.layers:
        before=x; x=layer(x,training=False); shape=tuple(int(v) for v in x.shape[1:]); macs=0
        if isinstance(layer,tf.keras.layers.Conv2D):
            kh,kw=layer.kernel_size; macs=int(np.prod(shape)*kh*kw*int(before.shape[-1])); receptive+=(kh-1)*jump; jump*=layer.strides[0]
        elif isinstance(layer,(tf.keras.layers.MaxPooling2D,tf.keras.layers.AveragePooling2D)):
            ph,pw=layer.pool_size; receptive+=(ph-1)*jump; jump*=layer.strides[0]
        elif isinstance(layer,tf.keras.layers.Dense): macs=int(before.shape[-1])*layer.units
        rows.append({"layer":layer.name,"output_shape":shape,"parameters":layer.count_params(),"activations":int(np.prod(shape)),"approx_MACs":macs,"receptive_field":receptive,"jump":jump})
    return pd.DataFrame(rows)


# %% [markdown]
# # Part A1 — TissueMNIST (grayscale)
#
# **What:** Load TissueMNIST through `medmnist`, normalize grayscale images, compare fixed filters with correlation/true convolution, then train a compact CNN.
#
# **Why:** A medical grayscale tensor makes channel handling and early feature extraction easy to inspect before moving to RGB.

# %%
try:
    import medmnist
    from medmnist import INFO, TissueMNIST
    tissue_available=True; print("medmnist available:",medmnist.__version__)
except Exception as exc:
    tissue_available=False; print("TissueMNIST unavailable; install medmnist in the linked venv to run A1:",exc)

# %%
if tissue_available:
    info=INFO["tissuemnist"]; tissue_classes=len(info["label"])
    def med_arrays(ds):
        images=np.asarray(ds.imgs,dtype=np.float32)/255.0; labels=np.asarray(ds.labels).reshape(-1).astype("int64")
        if images.ndim==3: images=images[...,None]
        return images,labels
    tissue_train,tissue_y_train=med_arrays(TissueMNIST(split="train",download=True)); tissue_val,tissue_y_val=med_arrays(TissueMNIST(split="val",download=True)); tissue_test,tissue_y_test=med_arrays(TissueMNIST(split="test",download=True))
    print(info["description"]); print(tissue_train.shape,tissue_val.shape,tissue_test.shape,"classes:",tissue_classes)
    assert tissue_train.shape[-1]==1 and tissue_train.dtype==np.float32
    show_maps([tissue_train[0,...,0]],[f"Original TissueMNIST label={tissue_y_train[0]}"],cmap="gray",cols=1,figsize=(4,4))
else:
    tissue_train=tissue_val=tissue_test=tissue_y_train=tissue_y_val=tissue_y_test=None

# %%
TISSUE_FILTERS={"horizontal Sobel":np.array([[-1,-2,-1],[0,0,0],[1,2,1]],np.float32),"vertical Sobel":np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32),"Laplacian":np.array([[0,1,0],[1,-4,1],[0,1,0]],np.float32),"5x5 blur":np.ones((5,5),np.float32)/25}
if tissue_available:
    sample_gray=tissue_train[0,...,0]; fixed={n:apply_odd_kernel(sample_gray,k) for n,k in TISSUE_FILTERS.items()}
    show_maps([sample_gray,*fixed.values()],["Original",*fixed.keys()],cmap="coolwarm",cols=3)
    kernel=TISSUE_FILTERS["horizontal Sobel"]; corr=apply_odd_kernel(sample_gray,kernel); conv=true_convolution(sample_gray,kernel)
    print("Correlation vs true convolution mean absolute difference:",float(np.mean(np.abs(corr-conv)))); assert corr.shape==conv.shape==sample_gray.shape
    show_maps([corr,conv],["Correlation (no flip)","True convolution (flipped kernel)"],cmap="coolwarm",cols=2)

# %% [markdown]
# ### Built-in Conv2D comparison
#
# **What:** Put the same fixed kernel into a Keras `Conv2D` layer and compare it with manual correlation.
#
# **Why:** TensorFlow calls the layer Conv2D, but its forward operation is cross-correlation unless weights are spatially flipped.

# %%
if tissue_available:
    conv_fixed=tf.keras.layers.Conv2D(1,3,padding="same",use_bias=False,kernel_initializer="zeros"); _=conv_fixed(tissue_train[:1])
    conv_fixed.set_weights([TISSUE_FILTERS["horizontal Sobel"][...,None,None]])
    keras_response=conv_fixed(tissue_train[:1]).numpy()[0,...,0]; manual_response=apply_odd_kernel(sample_gray,TISSUE_FILTERS["horizontal Sobel"],padding="constant")
    print("Built-in/manual max error:",float(np.max(np.abs(keras_response-manual_response)))); assert keras_response.shape==manual_response.shape

# %% [markdown]
# ### Small learned TissueMNIST CNN
#
# **What:** Train a short CNN and trace convolution, pooling, flatten, and latent activations.
#
# **Why:** Learned maps differ from fixed Sobel/blur responses because their weights are optimized for classification.

# %%
if tissue_available:
    tissue_model=tf.keras.Sequential([tf.keras.layers.Input(tissue_train.shape[1:]),tf.keras.layers.Conv2D(8,3,padding="same",activation="relu",name="tissue_conv1"),tf.keras.layers.MaxPooling2D(2,name="tissue_pool1"),tf.keras.layers.Conv2D(16,3,padding="same",activation="relu",name="tissue_conv2"),tf.keras.layers.MaxPooling2D(2,name="tissue_pool2"),tf.keras.layers.Flatten(name="tissue_flatten"),tf.keras.layers.Dense(32,activation="relu",name="tissue_latent"),tf.keras.layers.Dense(tissue_classes,activation="softmax",name="tissue_classifier")])
    tissue_model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]); tissue_model.fit(tissue_train,tissue_y_train,validation_data=(tissue_val,tissue_y_val),epochs=2,batch_size=128,verbose=2)
    print("Tissue test:",tissue_model.evaluate(tissue_test,tissue_y_test,verbose=0)); names=["tissue_conv1","tissue_pool1","tissue_conv2","tissue_pool2","tissue_flatten","tissue_latent"]; trace=tf.keras.Model(tissue_model.inputs[0],[tissue_model.get_layer(n).output for n in names]); acts=dict(zip(names,trace.predict(tissue_test[:1],verbose=0)))
    for n in names: print(n,acts[n].shape)
    show_maps([acts["tissue_conv1"][0,...,i] for i in range(8)],[f"learned conv1/{i}" for i in range(8)],cols=4)
    plt.figure(figsize=(10,3)); plt.bar(np.arange(acts["tissue_latent"].shape[-1]),acts["tissue_latent"][0]); plt.title("TissueMNIST learned latent vector (32D)"); plt.show()

# %%
if tissue_available:
    tissue_complexity=cnn_complexity(tissue_model,tissue_train.shape[1:]); display(tissue_complexity); assert tissue_complexity.parameters.sum()==tissue_model.count_params()
    display(pd.DataFrame({"concept":["hand-crafted","random","learned"],"source":["fixed Sobel/blur arrays","Conv2D initializer before fit","weights after supervised fit"],"purpose":["human prior","untrained baseline","task-adapted representation"]}))

# %% [markdown]
# # Part A2 — RGB `sons_of_anarchy.jpg`
#
# **What:** Decompose the relative RGB image path, apply custom filters per channel, compare random Conv2D responses, and optionally extract ImageNet features with MobileNetV2.
#
# **Why:** RGB filters combine three input channels, while a pretrained network demonstrates learned early/mid/late features at larger scale.

# %%
rgb=np.asarray(Image.open(IMAGE_PATH).convert("RGB").resize((224,224)),dtype=np.float32)/255.0; assert rgb.shape==(224,224,3) and np.isfinite(rgb).all()
show_maps([rgb,rgb[...,0],rgb[...,1],rgb[...,2]],["RGB original","red","green","blue"],cols=4,cmap=None)

# %%
rgb_demo={"red · horizontal Sobel":apply_odd_kernel(rgb[...,0],TISSUE_FILTERS["horizontal Sobel"]),"green · sharpen":apply_odd_kernel(rgb[...,1],np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],np.float32)),"blue · blur":apply_odd_kernel(rgb[...,2],TISSUE_FILTERS["5x5 blur"])}
show_maps(list(rgb_demo.values()),list(rgb_demo),cmap="coolwarm",cols=3)
random_rgb=tf.keras.layers.Conv2D(8,3,padding="same",activation="relu",kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(rgb[None]); print("Random (untrained) Conv2D maps:",random_rgb.shape)
show_maps([random_rgb[0,...,i] for i in range(8)],[f"random map {i}" for i in range(8)],cols=4)

# %%
try:
    mobile=tf.keras.applications.MobileNetV2(include_top=False,weights="imagenet",input_shape=(224,224,3)); mobile.trainable=False; print("Loaded MobileNetV2 ImageNet weights: learned pretrained features")
    mobile_layers=["block_1_expand_relu","block_6_expand_relu","out_relu"]; mobile_probe=tf.keras.Model(mobile.input,[mobile.get_layer(n).output for n in mobile_layers]); mobile_acts=mobile_probe.predict(rgb[None],verbose=0)
    for name,act in zip(mobile_layers,mobile_acts): print(name,act.shape); show_maps([act[0,...,i] for i in range(min(8,act.shape[-1]))],[f"{name} learned/{i}" for i in range(min(8,act.shape[-1]))],cols=4)
    mobile_latent=tf.keras.layers.GlobalAveragePooling2D()(mobile_acts[-1]).numpy()[0]; plt.figure(figsize=(12,3)); plt.plot(mobile_latent); plt.title(f"MobileNetV2 late latent vector ({mobile_latent.shape[0]}D)"); plt.show()
    display(pd.DataFrame({"stage":["early","mid","late"],"example_layer":mobile_layers,"meaning":["edges/textures","parts/repeated motifs","semantic channels"]}))
except Exception as exc:
    print("MobileNetV2 ImageNet feature extraction skipped: pretrained weights/model unavailable in this environment."); print("Fallback detail:",exc)

# %% [markdown]
# # Part B — LeNet-5 baseline and bounded tuning
#
# **What:** Preserve classic 32×32 padded-MNIST LeNet-5, search a small bounded KerasTuner space, retrain the winner from fresh weights, and report metrics/activations/latent vector.
#
# **Why:** The baseline anchors the historical architecture; bounded validation-only tuning demonstrates disciplined model selection.

# %%
(mnist_train_raw,mnist_y_train_raw),(mnist_test_raw,mnist_y_test)=tf.keras.datasets.mnist.load_data(); mnist_x_train_raw,mnist_x_val_raw,mnist_y_train,mnist_y_val=train_test_split(mnist_train_raw,mnist_y_train_raw,test_size=.1,stratify=mnist_y_train_raw,random_state=SEED)
def prep32(images): return np.pad(images.astype("float32")/255.0,((0,0),(2,2),(2,2)),mode="constant")[...,None]
mnist_x_train,mnist_x_val,mnist_x_test=map(prep32,[mnist_x_train_raw,mnist_x_val_raw,mnist_test_raw]); mnist_y_train,mnist_y_val,mnist_y_test=[x.astype("int64") for x in [mnist_y_train,mnist_y_val,mnist_y_test]]
assert mnist_x_train.shape[1:]==(32,32,1); show_maps([mnist_x_train[0,...,0]],[f"padded MNIST label={mnist_y_train[0]}"],cmap="gray",cols=1,figsize=(4,4))


# %%
def build_lenet():
    model=tf.keras.Sequential([tf.keras.layers.Input((32,32,1),name="input"),tf.keras.layers.Conv2D(6,5,activation="tanh",name="conv1"),tf.keras.layers.AveragePooling2D(2,name="pool1"),tf.keras.layers.Conv2D(16,5,activation="tanh",name="conv2"),tf.keras.layers.AveragePooling2D(2,name="pool2"),tf.keras.layers.Conv2D(120,5,activation="tanh",name="conv3"),tf.keras.layers.Flatten(name="flatten"),tf.keras.layers.Dense(84,activation="tanh",name="latent"),tf.keras.layers.Dense(10,activation="softmax",name="classifier")]); model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]); return model
baseline=build_lenet(); baseline.fit(mnist_x_train,mnist_y_train,validation_data=(mnist_x_val,mnist_y_val),epochs=2,batch_size=128,verbose=2); baseline_eval=baseline.evaluate(mnist_x_test,mnist_y_test,verbose=0); print("LeNet-5 baseline test loss/accuracy:",baseline_eval); display(cnn_complexity(baseline,(32,32,1)))

# %%
try:
    import keras_tuner as kt
    def lenet_builder(hp):
        filters=hp.Choice("filters",[6,12]); dense=hp.Choice("dense",[84,120]); lr=hp.Choice("learning_rate",[1e-3,3e-4]); m=tf.keras.Sequential([tf.keras.layers.Input((32,32,1),name="input"),tf.keras.layers.Conv2D(filters,5,activation="tanh",name="conv1"),tf.keras.layers.AveragePooling2D(2,name="pool1"),tf.keras.layers.Conv2D(16,5,activation="tanh",name="conv2"),tf.keras.layers.AveragePooling2D(2,name="pool2"),tf.keras.layers.Conv2D(120,5,activation="tanh",name="conv3"),tf.keras.layers.Flatten(name="flatten"),tf.keras.layers.Dense(dense,activation="tanh",name="latent"),tf.keras.layers.Dense(10,activation="softmax",name="classifier")]); m.compile(optimizer=tf.keras.optimizers.Adam(lr),loss="sparse_categorical_crossentropy",metrics=["accuracy"]); return m
    tuner=kt.RandomSearch(lenet_builder,objective="val_accuracy",max_trials=4,seed=SEED,overwrite=True,directory=str(ROOT/".keras_tuner_unified"),project_name="lenet5"); tuner.search(mnist_x_train,mnist_y_train,validation_data=(mnist_x_val,mnist_y_val),epochs=2,batch_size=128,verbose=0); best_hp=tuner.get_best_hyperparameters(1)[0]; print("Best bounded hyperparameters:",best_hp.values)
    tuned=lenet_builder(best_hp); tuned.fit(mnist_x_train,mnist_y_train,validation_data=(mnist_x_val,mnist_y_val),epochs=3,batch_size=128,verbose=2); tuned_eval=tuned.evaluate(mnist_x_test,mnist_y_test,verbose=0); pred=tuned.predict(mnist_x_test,verbose=0).argmax(1); report=classification_report(mnist_y_test,pred,output_dict=True,zero_division=0); display(pd.DataFrame([{ "model":"baseline","test_accuracy":baseline_eval[1],"parameters":baseline.count_params()},{"model":"tuned fresh retrain","test_accuracy":tuned_eval[1],"parameters":tuned.count_params(),"macro_f1":report["macro avg"]["f1-score"]}])); display(pd.DataFrame(confusion_matrix(mnist_y_test,pred)))
except Exception as exc:
    tuned=None; print("KerasTuner unavailable; bounded tuning skipped with explicit fallback:",exc)

# %%
selected=tuned if tuned is not None else baseline; trace_names=[n for n in ["conv1","pool1","conv2","pool2","conv3","flatten","latent","classifier"] if any(l.name==n for l in selected.layers)]; trace_model=tf.keras.Model(selected.inputs[0],[selected.get_layer(n).output for n in trace_names]); trace_acts=dict(zip(trace_names,trace_model.predict(mnist_x_test[:1],verbose=0)))
for n,a in trace_acts.items(): print(n,a.shape)
for n in ["conv1","pool1","conv2","pool2"]:
    a=trace_acts[n][0]; show_maps([a[...,i] for i in range(min(a.shape[-1],16))],[f"{n}/{i}" for i in range(min(a.shape[-1],16))],cols=4)
latent_name="latent" if "latent" in trace_acts else "flatten"; plt.figure(figsize=(12,3)); plt.bar(np.arange(trace_acts[latent_name].shape[-1]),trace_acts[latent_name][0]); plt.title(f"LeNet-5 latent vector: {latent_name} ({trace_acts[latent_name].shape[-1]}D)"); plt.show(); assert trace_acts["classifier"].shape==(1,10)

# %% [markdown]
# # Final learning and rubric/critique
#
# **What:** Summarize transferable lessons and audit the notebook against the requested evidence.
#
# **Why:** A critique table makes omissions visible instead of treating plots as proof by themselves.
#
# ### Additional learning
#
# 1. Correlation and true convolution differ by the spatial kernel flip; this matters when interpreting edge direction.
# 2. Pooling reduces spatial cost while increasing effective receptive field; later maps trade localization for abstraction.
# 3. Random kernels are not learned features: training aligns weights with labels.
# 4. RGB filters have one spatial slice per input channel and sum channel contributions into each output map.
# 5. A latent vector is a compact representation, not a class-probability vector.
# 6. Test metrics should be reported after model selection and fresh retraining.

# %%
rubric=pd.DataFrame([{"requirement":"A1 TissueMNIST grayscale + preprocessing + original image","evidence":"medmnist load, normalization assertions, original image plot"},{"requirement":"Odd-kernel helper + filters + correlation vs true convolution","evidence":"apply_odd_kernel, true_convolution, Sobel/blur/Laplacian plots"},{"requirement":"Built-in Conv2D comparison + trained CNN + traces","evidence":"fixed Conv2D error check; Tissue CNN activations/latent/complexity"},{"requirement":"A2 RGB image + custom filter demo","evidence":"relative sons_of_anarchy.jpg path; channel/custom-filter maps"},{"requirement":"MobileNetV2 early/mid/late + latent + fallback","evidence":"ImageNet load guarded; three probe stages and vector"},{"requirement":"Part B LeNet baseline + bounded tuning + fresh retrain","evidence":"classic model, max_trials=4, rebuilt tuned model, test report"},{"requirement":"Final learning and critique","evidence":"additional-learning text and this table"}]); display(rubric); print("Unified notebook checks complete.")
