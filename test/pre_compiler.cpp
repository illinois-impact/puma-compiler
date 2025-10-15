
#include <string>
#include <vector>
#include <assert.h>
#include "puma.h"
#include "models.h"
#include <map>
#include <functional>


typedef void (*FnPtr)(Model&);
std::map<std::string, FnPtr> func;

void init(){
    func["mlp_l4"] = mlp_l4;
    func["mlp_l5"] = mlp_l4;
    func["lstm"] = lstm;
    func["wlm_LSTM2048"] = wlm_LSTM2048;
    func["wlm_bigLSTM"] = wlm_bigLSTM;
    func["vgg19"] = vgg19;
    func["vgg16"] = vgg16;
    func["nmt_l5"] = nmt_l5;
    func["nmt_l3"] = nmt_l3;
    func["convmax"] = nmt_l3;
    func["conv_layer"] = conv_layer;
    func["simple"] = simple;
}

int main(int argc, char** argv){

    if(argc < 2){
        printf("Not enough args\n");
        return 0;
    }
    //initialize map
    init();


    std::string model_name = "";

    for(int i = 1; i < argc; i++){
        model_name += argv[i];
        model_name += "_";
    }
    model_name.pop_back(); //remove the last "_"
    // Ex : model1_model2...
    

    Model model = Model::create(model_name);
    //concatenate the selected models
    for(int i = 1; i < argc; i++){
        func[argv[i]](model);
    }
    

    // Compile
    model.compile();

    // Destroy model
    model.destroy();

    return 0;

}
