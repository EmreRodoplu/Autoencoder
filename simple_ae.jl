using Flux ,Plots ,MLDatasets ,Images

using Flux: flatten ,train! ,mse ,params

using Plots: plot

#Loading train data

x_train_raw , _ = MLDatasets.MNIST.traindata(Float32)

#Loading test data

x_test_raw , _ = MLDatasets.MNIST.testdata(Float32)

#view training data 

train_img = x_train_raw[:,:,1]

colorview(Gray ,train_img')

#view test data 

test_img = x_test_raw[:,:,1]

colorview(Gray ,test_img')

#Flatten train data

X_train = flatten(x_train_raw)

#Flatten test data

X_test = flatten(x_test_raw)

#Define model architecture

Encoder = Chain(
    BatchNorm(28*28),Dense(28*28 => 64 ,relu),
    BatchNorm(64),Dense(64 => 10),softmax
)

Decoder = Chain(
    BatchNorm(10),Dense(10 => 64 ,relu),
    BatchNorm(64),Dense(64 => 28*28,relu)
)

#Define loss function

loss(x,y) = mse(Decoder(Encoder(x)),y)

#Track parameters

ps = params(Encoder,Decoder)

#Select optimizer

learning_rate = 0.01

opt = ADAM(learning_rate)

#Train model

loss_history = []

train_epochs = []

epochs = 100

for epoch in 1:epochs
    #Model training
    train!(loss ,ps ,[(X_train,X_train)] ,opt)
    #Report
    train_loss = loss(X_train,X_train)
    push!(loss_history, train_loss)
    push!(train_epochs, epoch)
    println("Epoch = $epoch : Training Loss = $train_loss")
end

#Plotting data

plot(train_epochs ,loss_history ,xlabel = "Epoch" ,ylabel = "Loss")

#Choose a random index from test data

index = 100

#Displaying the chosen test data

colorview(Gray ,x_test_raw[:,:,index]')

#Predict the model

predict = Decoder(Encoder(X_test))

#Reshape the model to display

prediction = reshape(predict,(28,28,10000))

#Displaying the trained model

colorview(Gray ,prediction[:,:,index]')