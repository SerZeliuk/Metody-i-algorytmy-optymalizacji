include "disjoint.mzn";

enum ITEM;

array [int] of int: capacities;

array[ITEM] of int: profits;
array[ITEM] of int: weights;

var set of ITEM: knapsack_1;
var set of ITEM: knapsack_2;
var set of ITEM: knapsack_3;

constraint disjoint(knapsack_1, knapsack_2);
constraint disjoint(knapsack_1, knapsack_3);
constraint disjoint(knapsack_2, knapsack_3);

constraint sum (i in knapsack_1) (weights[i]) <= capacities[1];
constraint sum (i in knapsack_2) (weights[i]) <= capacities[2];
constraint sum (i in knapsack_3) (weights[i]) <= capacities[3];

solve maximize sum (i in knapsack_1) (profits[i]) + sum (i in knapsack_2) (profits[i]) + sum (i in knapsack_3) (profits[i]);

output ["Items count = \(length(ITEM))\n",
      "knapsack_1 = \(knapsack_1)\n",
      "profit = \(sum (i in knapsack_1) (profits[i]))\n",
      "knapsack_2 = \(knapsack_2)\n",
      "profit = \(sum (i in knapsack_2) (profits[i]))\n",
      "knapsack_3 = \(knapsack_3)\n",
      "profit = \(sum (i in knapsack_3) (profits[i]))\n"
];
