package pt.models

import org.apache.spark.sql.types.{IntegerType, StringType, StructType, TimestampType}

object SamsungHealthSchemas {

  val movementSchema: StructType = new StructType()
    .add("create_sh_ver", IntegerType, nullable = true)
    .add("start_time", TimestampType, nullable = true)
    .add("custom", StringType, nullable = true)
    .add("binning_data", StringType, nullable = true)
    .add("modify_sh_ver", IntegerType, nullable = true)
    .add("update_time", TimestampType, nullable = true)
    .add("create_time", TimestampType, nullable = true)
    .add("time_offset", StringType, nullable = true)
    .add("deviceuuid", StringType, nullable = true)
    .add("comment", StringType, nullable = true)
    .add("pkg_name", StringType, nullable = true)
    .add("end_time", TimestampType, nullable = true)
    .add("datauuid", StringType, nullable = true)

  val userProfileSchema: StructType = new StructType()
    .add("text_value", StringType, nullable = true)
    .add("create_sh_ver", IntegerType, nullable = true)
    .add("float_value", StringType, nullable = true)
    .add("modify_sh_ver", IntegerType, nullable = true)
    .add("update_time", TimestampType, nullable = true)
    .add("create_time", TimestampType, nullable = true)
    .add("long_value", StringType, nullable = true)
    .add("key", StringType, nullable = true)
    .add("blob_value", StringType, nullable = true)
    .add("int_value", StringType, nullable = true)
    .add("deviceuuid", StringType, nullable = true)
    .add("pkg_name", StringType, nullable = true)
    .add("double_value", StringType, nullable = true)
    .add("datauuid", StringType, nullable = true)

}
