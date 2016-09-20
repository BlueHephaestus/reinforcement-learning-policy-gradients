import itertools#Cartesian product stuff
import numpy as np
from scipy.optimize import minimize

import cho_base
from cho_base import HyperParameter

import policy_gradient_configurer
from policy_gradient_configurer import Configurer

#import sound_notifications
#import sms_notifications

"""
BEGIN USER INTERFACE FOR CONFIGURATION
"""
n_y = 20#Number of top y values we average over in our output
run_count = 3
run_decrement = 0#Amount we decrease the number of runs as we iterate. Will start at initial and never go < 1
output_training_cost = False#looking at avg timesteps, not this

#TODO: add this
global_config_count = 1

epochs = 400
timestep_n = 200

final_test_run = False

#Initialize our configurer
configurer = Configurer(epochs, timestep_n)

#The values we have until we optimize
initial_n = 0.1
initial_r_n = 6.0
initial_m = 10
initial_df = 0.95

for global_config_index in range(global_config_count):
    #Our default schedule to optimize a topology
    """MODIFIED BECAUSE OF INTERRUPTION"""
    hp_schedules = [
        [
            cho_base.HyperParameter(initial_m, initial_m, 0, .1, 1, "Mini Batch Size"),
            cho_base.HyperParameter(3.56, 3.56, 0.0, .1, .01, "Learning Rate"),
            cho_base.HyperParameter(.48, .48, 0.0, .1, .01, "Learning Rate Decay Rate"),
            cho_base.HyperParameter(initial_df, initial_df, 0, .1, 1, "Discount Factor"),
        ],
        [
            cho_base.HyperParameter(10, 100, 10, .1, 1, "Mini Batch Size"),
            cho_base.HyperParameter(initial_n, initial_n, 0, .1, .01, "Learning Rate"),
            cho_base.HyperParameter(initial_r_n, initial_r_n, 0, .1, .01, "Learning Rate Decay Rate"),
            cho_base.HyperParameter(initial_df, initial_df, 0, .1, 1, "Discount Factor"),
        ],
        [
            cho_base.HyperParameter(initial_m, initial_m, 0, .1, 1, "Mini Batch Size"),
            cho_base.HyperParameter(initial_n, initial_n, 0, .1, .01, "Learning Rate"),
            cho_base.HyperParameter(initial_r_n, initial_r_n, 0, .1, .01, "Learning Rate Decay Rate"),
            cho_base.HyperParameter(0.5, 1.0, .1, .1, 1, "Discount Factor"),
        ],
    ]

    """
    END USER INTERFACE FOR CONFIGURATION
    """
    #Global Config Optimization loop

    #For special configurations, like having three run at once with different initial values, I recommend simply changing them in the config file

    hp_mins = [hp.min for hp in hp_schedules[0]]#Set our initial values for this
    for hps in hp_schedules:
        #Update the old values for these with the new ones we obtained from last iteration.
        #and if this is the first iteration, no changes occur.
        for updated_hp_index, updated_hp in enumerate(hp_mins):
            if hps[updated_hp_index].step == 0:#Only do this if it's not being updated this iteration
                hps[updated_hp_index].min = updated_hp

        while True:
            #Normal optimization loop

            #Get our vectors to make cartesian product out of
            hp_vectors = [hp.get_vector() for hp in hps]#When we need the actual values
            print "New Optimization Initialized, Hyper Parameter Ranges for Config #%i are:" % (global_config_index)
            #sound_notifications.default_beeps()
            for hp_index, hp in enumerate(hp_vectors):
                print "\t%s: %s" % (hps[hp_index].label, ', '.join(map(str, hp)))

            #Check if we have set them all as constant, once we get our vectors again
            for hp in hp_vectors:
                if len(hp) > 1:
                    break
            else:
                """
                Since we have automated all the HP optimizations, these have been moved to the end of each global config run.
                print "Optimization Finished, Hyper Parameters are:"
                sms_message = "\nOptimization of Config #%i Finished, Hyper Parameters are:" % (global_config_index)#\n for header text

                for hp_index, hp in enumerate(hp_vectors):
                    print "\t%s: %f" % (hps[hp_index].label, hps[hp_index].min)
                    sms_message += "\n%s: %f" % (hps[hp_index].label, hps[hp_index].min)
                #Send text message notification
                if sms_alerts:
                    if sms_multiple_alerts:
                        sms_message += "\n\t-<3 C.H.O."
                        sms_notifications.send_sms(sms_message)
                    else:
                        sms_messages.append(sms_message)
                """


                if final_test_run:
                    #Feel free to disable this
                    print "Getting Final Optimized Resulting Values..."
                    config_avg_result = configurer.run_config(run_count, hp_vectors[0][0], 0.0, hp_vectors[1][0], 0.0, hp_vectors[2][0], hp_vectors[3][0])
                    print "And here are the Optimized Hyper Parameters again:"
                    for hp_index, hp in enumerate(hp_vectors):
                        print "\t%s: %f" % (hps[hp_index].label, hps[hp_index].min)


                hp_mins = [hp.min for hp in hps]
                break

            #Get cartesian product
            hp_cp = cho_base.cartesian_product(hp_vectors)

            hp_config_count = len(hp_cp)
            hp_cp_results = []#For the results in the cartesian product format, before averaging

            hp_ys = [np.copy(hp_vector).astype(float) for hp_vector in hp_vectors]
            coefs = []#For the coefficients of our quadratic regression of each hyper parameter

            #For our minimization
            #Uses the local min and max of each hp range
            bounds = [(hp.min, hp.max) for hp in hps]
            bounds = tuple(bounds)

            #Since we can just have 1s for each of our hps to plug in here.
            placeholder_hps = np.zeros_like(hps)
            #placeholder_hps = [0 for hp in hps]

            #Get our raw cp results/ys
            print "Computing Cartesian Product..."
            for hp_config_index, hp_config in enumerate(hp_cp):

                #Convert our np.float64 types to float
                hp_config = list(hp_config)
                hp_config[1:] = [float(hp) for hp in hp_config[1:]]

                #Execute configuration, get the average entry in the output_dict as a list of it's items
                config_avg_result = configurer.run_config(hp_config_index, len(hp_cp), run_count, hp_config[0], 0.0, hp_config[1], 0.0, hp_config[2], hp_config[3])
                #Get our highest n_y values from the respective output_type values
                config_y_vals = config_avg_result[:-n_y]#Get the last n_y values to look at.
                config_avg_y_val = np.mean(config_y_vals)
                #config_avg_y_val = sum(config_y_vals)/float(n_y)

                if not output_training_cost:
                    #So we make this into a find-the-minimum one if it's looking at accuracy, which we want to be higher
                    config_avg_y_val = timestep_n - config_avg_y_val

                #Add our result to each of our configs in hp_results
                hp_cp_results.append(config_avg_y_val)

            #hp_ys is used to get the average output using our hp caused, so if we had 3 mini batches and 3 regularization rates,
            #our associated hp_y value for our first mini batch size will be the average over the 3 runs that used the first mini batch size.
            #This is where we get those averages.
            print "Averaging respective Hyper Parameter Output..."
            for hp_index, hp in enumerate(hp_vectors):
                for hp_val_index, hp_val in enumerate(hp):
                    hp_val_output_sum = 0
                    n_hp_val = 0
                    for config_index, config in enumerate(hp_cp):
                        if hp_val == config[hp_index]:#
                            hp_val_output_sum += hp_cp_results[config_index]
                            n_hp_val += 1
                    hp_ys[hp_index][hp_val_index] = hp_val_output_sum/float(n_hp_val)

            #Get our coefficients by doing a quadratic regression on each of our average output for each hyper parameter set
            print "Obtaining Quadratic Regression Coefficients..."
            for hp_index, hp in enumerate(hp_vectors):
                if len(hp) > 1:
                    coefs.append(np.polynomial.polynomial.polyfit(hp, hp_ys[hp_index], 2))
                else:
                    coefs.append([hp[0], 0, 0])

            print "Coefficients are %s" % (', '.join(map(str, coefs)))

            print "Computing Minimum of Multivariable Hyper Parameter Function..."

            #Our main function to minimize once we have our coefficients
            hp_function = lambda hps: sum([coef[0] + coef[1]*hp + coef[2]*hp**2 for coef, hp in zip(coefs, hps)])#Why the fuck are quadratic regression coefficient orders backwards

            res = minimize(hp_function, placeholder_hps, bounds=bounds, method='TNC', tol=1e-10, options={'xtol': 1e-8, 'disp': False})

            #Now our res.x are our new center point values
            center_points = res.x

            print "Minimum values are: %s" % (', '.join(map(str, center_points)))
            #print center_points

            print "Computing new Hyper Parameter Optimization Ranges..."
            for hp_index, center_point in enumerate(center_points):
                if len(hp_vectors[hp_index]) > 1:
                    step = hps[hp_index].step
                    step_decrease_factor = hps[hp_index].step_decrease_factor
                    stop_threshold = hps[hp_index].stop_threshold
                    new_step = step*step_decrease_factor

                    #print new_step, stop_threshold
                    if new_step < stop_threshold:
                        #Time to mark this value as final and stop modifying.
                        #This means we no longer update it, we just replace the min and max with our center point,
                        #and make the step 0. Just as we do with dependent variables at the start
                        new_min = center_point
                        new_max = center_point
                        new_step = 0
                    else:
                        #We get our inclusive range, i.e if center point is 19.14, 
                        #We'd get 14.14, and 24.14. Then we round up and round down respectively,
                        #to get 15 and 24
                        new_min = cho_base.step_roundup(center_point-(step*.5), new_step)
                        new_max = cho_base.step_rounddown(center_point+(step*.5), new_step)

                    #We update with our new params if an independent hyper parameter
                    hps[hp_index].min = new_min
                    hps[hp_index].max = new_max
                    hps[hp_index].step = new_step
                    #print new_min, new_max

            #Decrement our run number
            if run_count > 1: 
                run_count -= run_decrement

    sms_message = "\nOptimization of Config #%i Finished, Hyper Parameters are:" % (global_config_index)#\n for header text
    for hp_index, hp in enumerate(hp_vectors):
        print "\t%s: %f" % (hps[hp_index].label, hps[hp_index].min)
        sms_message += "\n%s: %f" % (hps[hp_index].label, hps[hp_index].min)

    #Send text message notification
    if sms_alerts:
        if sms_multiple_alerts:
            sms_message += "\n\t-<3 C.H.O."
            sms_notifications.send_sms(sms_message)
        else:
            sms_messages.append(sms_message)

        
if sms_alerts and not sms_multiple_alerts:
    #Send our big message
    sms_messages[-1] += "\n\t-<3 C.H.O."
    for sms_message in sms_messages:
        sms_notifications.send_sms(sms_message)
        print sms_message
