#!/usr/bin/env python
# coding: utf-8

# In[68]:


import sqlite3 as sql
import pandas as pd


# In[69]:


conn = sql.connect('assignment.db')
c = conn.cursor()


# In[70]:


data = pd.read_csv("multilingual-age.csv",header = 4)


# In[71]:


c.execute("DROP TABLE IF EXISTS place ;")

c.execute("""CREATE TABLE place(
               s_code INTEGER , 
               d_code INTEGER ,
               area TEXT,
               PRIMARY KEY (s_code , d_code , area));""")

lst = data.values.tolist()
for i in range(len(data)):
    try : c.execute("""INSERT INTO place VALUES(?,?,?);""",(int(lst[i][0]),int(lst[i][1]),lst[i][2]))
    except : pass
conn.commit()


# In[72]:


c.execute("DROP TABLE IF EXISTS biling_age ;")
c.execute("DROP TABLE IF EXISTS triling_age ;")

c.execute("""CREATE TABLE biling_age(
                area TEXT , 
                type TEXT ,
                age TEXT ,
                person BIGINT,
                male BIGINT,
                female BIGINT);""")

c.execute("""CREATE TABLE triling_age(
                area TEXT , 
                type TEXT ,
                age TEXT ,
                person BIGINT,
                male BIGINT,
                female BIGINT);""")

lst = data.values.tolist()


for i in range(len(lst)):
    c.execute("""INSERT INTO biling_age VALUES(?,?,?,?,?,?);""",(lst[i][2],lst[i][3],lst[i][4],int(lst[i][5]),int(lst[i][6]),int(lst[i][7])))
    c.execute("""INSERT INTO triling_age VALUES(?,?,?,?,?,?);""",(lst[i][2],lst[i][3],lst[i][4],int(lst[i][8]),int(lst[i][9]),int(lst[i][10])))
conn.commit()    


# In[73]:


data1 = pd.read_csv("multilingual-education.csv",header = 4)


# In[74]:


c.execute("DROP TABLE IF EXISTS biling_edu ;")
c.execute("DROP TABLE IF EXISTS triling_edu ;")

c.execute("""CREATE TABLE biling_edu(
                area TEXT , 
                type TEXT ,
                edu TEXT ,
                person BIGINT,
                male BIGINT,
                female BIGINT);""")

c.execute("""CREATE TABLE triling_edu(
                area TEXT , 
                type TEXT ,
                edu TEXT ,
                person BIGINT,
                male BIGINT,
                female BIGINT);""")

lst = data1.values.tolist()

for i in range(len(lst)):
    c.execute("""INSERT INTO biling_edu VALUES(?,?,?,?,?,?);""",(lst[i][2],lst[i][3],lst[i][4],int(lst[i][5]),int(lst[i][6]),int(lst[i][7])))
    c.execute("""INSERT INTO triling_edu VALUES(?,?,?,?,?,?);""",(lst[i][2],lst[i][3],lst[i][4],int(lst[i][8]),int(lst[i][9]),int(lst[i][10])))
conn.commit()    


# In[75]:


data2 = pd.read_csv("age-education.csv",header = 6)


# In[136]:


c.execute("DROP TABLE IF EXISTS age_edu ;")
c.execute("DROP TABLE IF EXISTS age_edu_classified ;")

c.execute("""CREATE TABLE age_edu(
               area TEXT,
               type TEXT,
               age TEXT,
               total_p BIGINT , total_m BIGINT , total_f BIGINT,
               gender_skew FLOAT);""")

c.execute("""CREATE TABLE age_edu_classified(
               area TEXT,
               type TEXT,
               age TEXT,
               illt_p BIGINT , illt_m BIGINT , illt_f BIGINT,
               lit_p BIGINT , lit_m BIGINT  , lit_f BIGINT,
               edu_lit_p BIGINT , edu_lit_m BIGINT , edu_lit_f BIGINT,
               bel_primary_p BIGINT , bel_primary_m BIGINT , bel_primary_f BIGINT,
               primary_p BIGINT , primary_m BIGINT , primary_f BIGINT,
               middle_p BIGINT , middle_m BIGINT , middle_f BIGINT,
               secondary_p BIGINT , secondary_m BIGINT , secondary_f BIGINT,
               sen_secondary_p BIGINT , sen_secondary_m BIGINT , sen_secondary_f BIGINT,
               nonTechDip_p BIGINT , nonTechDip_m BIGINT , nonTechDip_f BIGINT,
               diploma_p BIGINT , diploma_m BIGINT , diploma_f BIGINT,
               graduate_p BIGINT , graduate_m BIGINT , graduate_f BIGINT,
               unclass_p BIGINT , unclass_m BIGINT , unclass_f BIGINT);""")

lst = data2.values.tolist()

for i in range(len(lst)):
    if 'State' in lst[i][3] :
        a = lst[i][3][8:]
    else:
        a = lst[i][3]
    value1 = [a,lst[i][4],lst[i][5]]
    value2 = [a,lst[i][4],lst[i][5]]
    for j in range(6,9):
        value1.append(int(lst[i][j]))
    for j in range(9,45):
        value2.append(int(lst[i][j]))
    value1.append(0)
    c.execute("INSERT INTO age_edu VALUES(?,?,?,?,?,?,?);",value1)
    c.execute("INSERT INTO age_edu_classified VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);",value2)
      
conn.commit() 


# In[137]:


c.execute("""UPDATE age_edu SET age = 'Total' WHERE age = 'All ages' ;""")
conn.commit()
c.execute("""UPDATE age_edu SET gender_skew = ((1.0 * total_m )/ total_f) WHERE ((1.0 * total_m )/ total_f) > 1 ;""")
conn.commit()
c.execute("""UPDATE age_edu SET gender_skew = ((1.0 * total_f )/ total_m) WHERE ((1.0 * total_m )/ total_f) <= 1 ;""")
conn.commit()


# In[138]:



#question 1
c.execute("""SELECT area , (((1.0 *person) / total_p) * 100) 
             FROM triling_age JOIN age_edu USING (area , type ,age) 
             WHERE area != 'INDIA' and type = 'Total' and age = 'Total' 
             ORDER BY (((1.0 *person) / total_p) * 100) DESC;""")
print("Order the states and union territories by the percentage of trilingual population in them :")
print()
print(c.fetchall())
print()


# In[160]:


#question 2
c.execute("""CREATE VIEW multiling_age1 AS
             SELECT age , biling_age.person+triling_age.person AS multi_person
             FROM biling_age JOIN triling_age USING(area , type , age)
             WHERE type = 'Total' and area = 'INDIA' and age != 'Total' ;""")
c.fetchall()


# In[161]:


c.execute("""ALTER TABLE age_edu ADD totalPep TEXT;""")


# In[162]:


c.execute("""UPDATE age_edu SET totalPep = 
                    CASE(age_edu.age)
                    WHEN '0-6'   THEN '0'
                         WHEN '7'     THEN '5-9'
                         WHEN '8'     THEN '5-9'
                         WHEN '9'     THEN '5-9'
                         WHEN '10'    THEN '10-14'
                         WHEN '11'    THEN '10-14'
                         WHEN '12'    THEN '10-14'
                         WHEN '13'    THEN '10-14'
                         WHEN '14'    THEN '10-14'
                         WHEN '15'    THEN  '15-19'
                         WHEN '16'    THEN '15-19'
                         WHEN '17'    THEN '15-19'
                         WHEN '18'    THEN '15-19'
                         WHEN '19'    THEN '15-19'
                         WHEN '20-24'    THEN '20-24'
                         WHEN '25-29'    THEN '25-29'
                         WHEN '30-34'    THEN '30-49'
                         WHEN '35-39'    THEN '30-49'
                         WHEN '40-44'    THEN '30-49'
                         WHEN '45-49'    THEN '30-49'
                         WHEN '50-54'    THEN '50-69'
                         WHEN '55-59'    THEN '50-69'
                         WHEN '60-64'    THEN '50-69'
                         WHEN '65-69'    THEN '50-69'
                         WHEN '70-74'    THEN '70+'
                         WHEN '75-79'    THEN '70+'
                         WHEN '80+'      THEN '70+'
                         ELSE 'Age not stated'
                      END;""")
conn.commit()


# In[163]:


c.execute("""CREATE VIEW all_pop AS
             SELECT totalPep as age , sum(total_p) AS people
             FROM age_edu
             WHERE area = 'INDIA' and type = 'Total' and age != 'Total' and age != '0-6'
             GROUP BY totalPep;""")


# In[164]:


c.execute("""SELECT age
             FROM all_pop , multiling_age1 
             WHERE all_pop.totalPep = multiling_age1.age and 1.0 * multi_person / people = (
                    SELECT max(1.0 * multi_person / people )
                    FROM all_pop , multiling_age1 
                    WHERE all_pop.totalPep = multiling_age1.age) ;""")
print(" Which age group has the largest percentage of multi-lingual population")
print()
print(c.fetchall())
print()


# In[116]:



#question 3
c.execute("""SELECT age 
             FROM age_edu
             WHERE area = 'INDIA' and type = 'Total' and age != 'Total' and gender_skew = (
                   SELECT max(gender_skew)
                   FROM age_edu
                   WHERE area = 'INDIA' and type = 'Total' and age != 'Total');""")

print(" For the whole of India, which age group depicts the most skewed gender ratio :")
print()
print(c.fetchall())
print()


# In[117]:



#question 4
c.execute("""CREATE VIEW IF NOT EXISTS temp AS
                   SELECT area , type , age , triling_age.person + biling_age.person AS multiling  
                   FROM biling_age JOIN triling_age USING(area , type , age)
                   WHERE area = 'INDIA' and type = 'Total' and age = 'Total'; """)

c.execute("""  SELECT total_p - multiling
               FROM  temp JOIN age_edu USING (area ,type , age)
               WHERE area = 'INDIA' and type = 'Total' and age = 'Total'; """)
print("Find the total number of people in India who speaks only one language :")
print()
print(c.fetchall())
print()


# In[118]:


c.execute("""ALTER TABLE age_edu ADD avgAge INT DEFAULT 0 ;""")
c.execute("""UPDATE age_edu SET avgAge = 
                      CASE (age_edu.age)
                         WHEN '0-6'   THEN 3
                         WHEN '7'     THEN 7
                         WHEN '8'     THEN 8
                         WHEN '9'     THEN 9
                         WHEN '10'    THEN 10
                         WHEN '11'    THEN 11
                         WHEN '12'    THEN 12
                         WHEN '13'    THEN 13
                         WHEN '14'    THEN 14
                         WHEN '15'    THEN 15
                         WHEN '16'    THEN 16
                         WHEN '17'    THEN 17
                         WHEN '18'    THEN 18
                         WHEN '19'    THEN 19
                         WHEN '20-24'    THEN 22
                         WHEN '25-29'    THEN 27
                         WHEN '30-34'    THEN 32
                         WHEN '35-39'    THEN 37
                         WHEN '40-44'    THEN 42
                         WHEN '45-49'    THEN 47
                         WHEN '50-54'    THEN 52
                         WHEN '55-59'    THEN 57
                         WHEN '60-64'    THEN 62
                         WHEN '65-69'    THEN 67
                         WHEN '70-74'    THEN 72
                         WHEN '75-79'    THEN 77
                         WHEN '80+'      THEN 85
                         ELSE 0
                      END;""")
conn.commit()


# In[119]:



#question 5
c.execute("""CREATE VIEW summation AS
             SELECT area , SUM(avgAge*total_p) AS val
             FROM  age_edu
             WHERE area != 'INDIA' and type = 'Total' and age != 'Total' and age != 'Age not stated' 
             GROUP BY area ;""")

c.execute("""CREATE VIEW allPeople AS
             SELECT area , total_p
             FROM age_edu
             WHERE area != 'INDIA' and type = 'Total' and age = 'Total' 
             GROUP BY area ;""")


# In[120]:


c.execute("""SELECT area 
             FROM (summation JOIN allPeople USING (area)) as main
             WHERE 1.0*val / total_p = (
                    SELECT max(1.0*val / total_p)
                    FROM summation JOIN allPeople USING (area)) ;""")

print("""In which state or union territory is the average age the highest? (For the age-group 0-6, assume the average age to be 3, while for the age group 80+, assume the average age to be 85. For the groups 20-24, etc., assume the average age to be 22, etc :""")
print()
print(c.fetchall())
print()


# In[ ]:




