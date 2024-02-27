Lasso profiles: 

    Four most important features on eicu: ['numeric__sbp', 'numeric__na', 'numeric__temp', 'numeric__hr']
    Four most important features on mimic: ['numeric__temp', 'numeric__age', 'numeric__height', 'numeric__hr']
    Four most important features on miiv: ['numeric__map', 'numeric__mch', 'numeric__sbp', 'numeric__hr']
    Four most important features on hirid: ['numeric__sbp', 'numeric__ckmb', 'numeric__mch', 'numeric__hr']
    Four most important features on all: ['numeric__na', 'numeric__temp', 'numeric__height', 'numeric__hr']

DSL profiles: (No categorical in individual effects after preprocessing?)

    common effects:  ['numeric__temp', 'numeric__hr', 'categorical__sex_Female', 'categorical__sex_Male']
    eicu effects:  ['eicu numeric__po2', 'eicu numeric__alb', 'eicu numeric__temp', 'eicu numeric__fio2']
    mimic effects:  ['mimic numeric__dbp', 'mimic numeric__map', 'mimic numeric__age', 'mimic numeric__height']
    miiv effects:  ['miiv numeric__sbp', 'miiv numeric__map', 'miiv numeric__mcv', 'miiv numeric__mch']
    hirid effects:  ['hirid numeric__dbp', 'hirid categorical__sex_Male', 'hirid numeric__bun', 'hirid numeric__mch']