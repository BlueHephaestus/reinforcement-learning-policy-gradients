#So we can import without being there
import sys

#For Cho's use optimizing
import policy_grads6

class Configurer(object):
    def __init__(self, epochs, output_types, output_training_cost, output_training_accuracy, output_validation_accuracy, output_test_accuracy, archive_dir):
        self.epochs = epochs

    def run_config(self, run_count, mini_batch_size, learning_rate, optimization, optimization_term1, optimization_term2, regularization_rate, p_dropout, global_config_index, config_index, config_count ):#Last two are for progress

        #After all runs have executed
        #If there were more than one runs for this configuration, we average them all together for a new one
        #We do this by looping through all our y values for each epoch, and doing our usual mean calculations
        if run_count > 1:
            output_dict[run_count+1] = {}#For our new average entry
            for j in range(self.epochs):
                output_dict[run_count+1][j] = []#For our new average entry
                for o in range(self.output_types):
                    avg = sum([output_dict[r][j][o] for r in range(run_count)]) / run_count
                    output_dict[run_count+1][j].append(avg)
            return output_dict[run_count+1]#Return our average end result
        else:
            return output_dict[0]#Return our run, since we only did one.
