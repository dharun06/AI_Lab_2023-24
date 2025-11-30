# Ex.No: 10  Logic Programming –  Simple queries from facts and rules
### DATE: 06/09/2025                                                                         
### REGISTER NUMBER : 212223060051
### AIM: 
To write a prolog program to find the answer of query. 
###  Algorithm:
 Step 1: Start the program <br> 
 Step 2: Convert the sentence into First order Logic  <br> 
 Step 3:  Convert the sentence into Horn clause form  <br> 
 Step 4: Add rules and predicates in a program   <br> 
 Step 5:  Pass the query to program. <br> 
 Step 6: Prolog interpreter shows the output and return answer. <br> 
 Step 8:  Stop the program.
### Program:
### Task 1:
Construct the FOL representation for the following sentences <br> 
1.	John likes all kinds of food.  <br> 
2.	Apples are food.  <br> 
3.	Chicken is a food.  <br> 
4.	Sue eats everything Bill eats. <br> 
5.	 Bill eats peanuts  <br> 
   Convert into clause form and Prove that John like Apple by using Prolog. <br> 
### Program:

```
likes(john,X):-food(X).
food(apple).
food(chicken).
food(peanuts).
eats(bill,X):-eats(sue,X).
eats(bill,peanuts).
```


### Output:

<img width="920" height="69" alt="image" src="https://github.com/user-attachments/assets/9f607965-f30b-4ce7-b67f-5afdfc61d916" />



### Task 2:
Consider the following facts and represent them in predicate form: <br>              
1.	Steve likes easy courses. <br> 
2.	Science courses are hard. <br> 
3. All the courses in Have fun department are easy <br> 
4. BK301 is Have fun department course.<br> 
Convert the facts in predicate form to clauses and then prove by resolution: “Steve likes BK301 course”<br> 

### Program:
```
like(steve,X):-easycourse(X).
hard(sciencecourse).
easycourse(X):-course(X,fundept).
course(bk301,fundept).
```


### Output:
<img width="920" height="71" alt="image" src="https://github.com/user-attachments/assets/29f5ca87-ab48-41db-8c6a-932238ce7b31" />


### Task 3:
Consider the statement <br> 
“This is a crime for an American to sell weapons to hostile nations. The Nano , enemy of America has some missiles and its missiles were sold it by Colonal West who is an American” <br> 
Convert to Clause form and prove west is criminal by using Prolog.<br> 
### Program:
```
crime(X):-american(X),weapon(Y),sells(X,Y,Z),hostile(X,Z).
hostile(X,Z):-enemy(Z,X).
enemy(nano,west).
american(west).
weapon(Y):-missile(Y).
missile(m1).
owns(nano,m1).
sells(west,Y,nano):-owns(nano,Y),missile(Y).
```


### Output:
<img width="924" height="84" alt="image" src="https://github.com/user-attachments/assets/53a70a29-f8b8-475a-aed1-affee561c3d4" />


### Result:
Thus the prolog programs were executed successfully and the answer of query was found.
