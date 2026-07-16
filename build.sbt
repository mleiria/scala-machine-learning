ThisBuild / scalaVersion := "2.12.18"

lazy val root = (project in file("."))
  .settings(
    name := "scala-machine-learning",
    version := "0.1.22",
    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked",
      "-Xlint"
    ),
    javaOptions ++= Seq(
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
      "--add-opens=java.base/java.util.logging=ALL-UNNAMED",
      "--add-opens=java.base/java.util.zip=ALL-UNNAMED",
      "--add-opens=java.base/java.util.regex=ALL-UNNAMED",
      "--add-opens=java.base/java.net=ALL-UNNAMED",
      "--add-opens=java.base/java.io=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED"
    ),
    fork := true
  )
  val sparkVersion = "3.5.8"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "2.1.0",
  "org.scalanlp" %% "breeze-natives" % "2.1.0",
  "commons-logging" % "commons-logging" % "1.2",
  "org.slf4j" % "slf4j-simple" % "1.7.32",
  "org.scalatest" %% "scalatest" % "3.2.9" % "test",
  "junit" % "junit" % "4.10" % "test",
  "com.google.guava" % "guava" % "29.0-jre",
  "org.apache.httpcomponents" % "httpclient" % "4.5.14",
  "org.mongodb" % "mongo-java-driver" % "2.10.1",
  "io.github.cibotech" %% "evilplot" % "0.9.0",
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql"  % sparkVersion
)