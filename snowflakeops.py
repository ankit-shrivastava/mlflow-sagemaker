from snowflake import connector


def snowflake_connection(account, username, password, role, warehouse, database, schema):
    connection = None

    try:
        connection = connector.connect(
            account=account,
            user=username,
            password=password,
            role=role,
            warehouse=warehouse,
            database=database,
            schema=schema
        )
        print("SNOWFLAKE CONNECTION SUCCESSFUL")
    except Exception as e:
        print("CONNECTION TO SNOWFLAKE FAILED", e)
        raise e
    return connection


def execute_query(connection, sql_query):
    if connection:
        try:
            cursor = connection.cursor()
            if cursor:
                cursor.execute(sql_query)
                try:
                    results = cursor.fetchall()
                    return results
                except:
                    pass
        except Exception as e:
            print("Failed to execute query", e)
            raise e
        finally:
            if cursor:
                cursor.close()
