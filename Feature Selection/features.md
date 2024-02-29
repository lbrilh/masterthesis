*HR*

Lasso profiles without sex: 

    Six most important features on eicu: ['numeric__sbp', 'numeric__na', 'numeric__temp', 'numeric__hr', 'categorical__sex_Female', 'categorical__sex_Male']
    Six most important features on mimic: ['numeric__temp', 'numeric__age', 'numeric__height', 'numeric__hr', 'categorical__sex_Female', 'categorical__sex_Male']
    Six most important features on miiv: ['numeric__map', 'numeric__mch', 'numeric__sbp', 'numeric__hr', 'categorical__sex_Male', 'categorical__sex_Female']
    Six most important features on hirid: ['numeric__sbp', 'numeric__ckmb', 'numeric__mch', 'numeric__hr', 'categorical__sex_Male', 'categorical__sex_Female']
    Six most important features on all: ['numeric__na', 'numeric__temp', 'numeric__height', 'numeric__hr', 'categorical__sex_Female', 'categorical__sex_Male']

DSL profiles incl. sex: (preprocessed all at once)

    common effects:  ['numeric__temp', 'numeric__hr', 'categorical__sex_Female', 'categorical__sex_Male']
    eicu effects:  ['eicu numeric__po2', 'eicu numeric__alb', 'eicu numeric__temp', 'eicu numeric__fio2']
    mimic effects:  ['mimic numeric__dbp', 'mimic numeric__map', 'mimic numeric__age', 'mimic numeric__height']
    miiv effects:  ['miiv numeric__sbp', 'miiv numeric__map', 'miiv numeric__mcv', 'miiv numeric__mch']
    hirid effects:  ['hirid numeric__dbp', 'hirid categorical__sex_Male', 'hirid numeric__bun', 'hirid numeric__mch']

DSL profiles incl. sex: (preprocessed all individually)
    common effects:  ['numeric__temp', 'numeric__hr', 'categorical__sex_Male', 'categorical__sex_Female']
    eicu effects:  ['eicu numeric__ph', 'eicu numeric__bun', 'eicu numeric__fio2', 'eicu numeric__temp']
    mimic effects:  ['mimic numeric__height', 'mimic numeric__hr', 'mimic categorical__sex_Female', 'mimic categorical__sex_Male']
    miiv effects:  ['miiv numeric__sbp', 'miiv numeric__mcv', 'miiv numeric__map', 'miiv numeric__mch']
    hirid effects:  ['hirid numeric__ckmb', 'hirid numeric__alt', 'hirid numeric__mch', 'hirid categorical__sex_Male']


*MAP* 

Lasso profiles: 
    
    Six most important features on eicu: ['numeric__temp', 'numeric__sbp', 'numeric__dbp', 'numeric__map', 'categorical__sex_Female', 'categorical__sex_Male']
    Six most important features on mimic: ['numeric__mchc', 'numeric__mcv', 'numeric__mch', 'numeric__map', 'categorical__sex_Female', 'categorical__sex_Male']
    Six most important features on miiv:  ['numeric__mchc', 'numeric__mcv', 'numeric__mch', 'numeric__map', 'categorical__sex_Female', 'categorical__sex_Male']
    Six most important features on hirid: ['numeric__hr', 'numeric__mch', 'numeric__dbp', 'numeric__map', 'categorical__sex_Male', 'categorical__sex_Female']
    Six most important features on all: ['numeric__temp', 'numeric__dbp', 'numeric__sbp', 'numeric__map', 'categorical__sex_Female', 'categorical__sex_Male']


DSL profiles incl sex: (preprocessed all at once)

    common effects: ['numeric__resp', 'numeric__map', 'categorical__sex_Female', 'categorical__sex_Male']
    eicu effects: ['eicu numeric__sbp', 'eicu categorical__sex_Female', 'eicu numeric__dbp', 'eicu numeric__map']
    mimic effects: ['mimic numeric__be', 'mimic numeric__mchc', 'mimic numeric__mcv', 'mimic numeric__mch']
    miiv effects: ['miiv numeric__bicar', 'miiv numeric__mchc', 'miiv numeric__mcv', 'miiv numeric__mch']
    hirid effects: ['hirid numeric__mchc', 'hirid categorical__sex_Male', 'hirid numeric__mch', 'hirid categorical__sex_Female']