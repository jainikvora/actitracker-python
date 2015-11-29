import mysql.connector as mysql
import pandas as pd


class ActitrackerData:
    @staticmethod
    def get_features():
        try:
            conn = mysql.connect(user='root', password='admin', host='localhost', database='actitracker')
        except mysql.connector.Error as err:
            print(err)
            conn.close()
        cursor = conn.cursor()

        query = "SELECT mean0, mean1, mean2, variance0, variance1, variance2, avgabsdiff0, avgabsdiff1, " \
                "avgabsdiff2, resultant, avgtimepeak FROM activity_with_features_3"

        cursor.execute(query)
        columns = tuple([d[0].decode('utf8') for d in cursor.description])
        records = []
        for record in cursor:
            records.append(dict(zip(columns, record)))

        data = pd.DataFrame(records)
        cursor.close()
        return data

    @staticmethod
    def get_lables():
        try:
            conn = mysql.connect(user='root', password='admin', host='localhost', database='actitracker')
        except mysql.connector.Error as err:
            print(err)
            conn.close()
        cursor = conn.cursor()
        lable_query = "SELECT lable FROM activity_with_features_3"

        cursor.execute(lable_query)
        columns = tuple([d[0].decode('utf8') for d in cursor.description])
        lables = []
        for lable in cursor:
            lables.append(dict(zip(columns, lable)))

        lable_data = pd.DataFrame(lables)
        cursor.close()
        return lable_data
