import utils.Model as Model

def main():
    models = Model.TrainingModels()
    models.set_preload_text()
    classes =['Life_Cycle','News','Application Brief','Product Brief','Package Drawing','PCN','Package Brief','Others','Datasheet','Environmental']
 
    class_weights = {i:5 for i in classes}
    models.train_multi_level(sampling=0,load_trained=0,Classes_sample=classes,class_weights=class_weights)
 

if __name__ == "__main__":
    main()