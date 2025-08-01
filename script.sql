USE [master]
GO
/****** Object:  Database [dataset]    Script Date: 31/07/2025 11:02:10 ******/
CREATE DATABASE [dataset]
 CONTAINMENT = NONE
 ON  PRIMARY 
( NAME = N'dataset', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER\MSSQL\DATA\dataset.mdf' , SIZE = 8192KB , MAXSIZE = UNLIMITED, FILEGROWTH = 65536KB )
 LOG ON 
( NAME = N'dataset_log', FILENAME = N'C:\Program Files\Microsoft SQL Server\MSSQL16.MSSQLSERVER\MSSQL\DATA\dataset_log.ldf' , SIZE = 8192KB , MAXSIZE = 2048GB , FILEGROWTH = 65536KB )
 WITH CATALOG_COLLATION = DATABASE_DEFAULT, LEDGER = OFF
GO
ALTER DATABASE [dataset] SET COMPATIBILITY_LEVEL = 160
GO
IF (1 = FULLTEXTSERVICEPROPERTY('IsFullTextInstalled'))
begin
EXEC [dataset].[dbo].[sp_fulltext_database] @action = 'enable'
end
GO
ALTER DATABASE [dataset] SET ANSI_NULL_DEFAULT OFF 
GO
ALTER DATABASE [dataset] SET ANSI_NULLS OFF 
GO
ALTER DATABASE [dataset] SET ANSI_PADDING OFF 
GO
ALTER DATABASE [dataset] SET ANSI_WARNINGS OFF 
GO
ALTER DATABASE [dataset] SET ARITHABORT OFF 
GO
ALTER DATABASE [dataset] SET AUTO_CLOSE OFF 
GO
ALTER DATABASE [dataset] SET AUTO_SHRINK OFF 
GO
ALTER DATABASE [dataset] SET AUTO_UPDATE_STATISTICS ON 
GO
ALTER DATABASE [dataset] SET CURSOR_CLOSE_ON_COMMIT OFF 
GO
ALTER DATABASE [dataset] SET CURSOR_DEFAULT  GLOBAL 
GO
ALTER DATABASE [dataset] SET CONCAT_NULL_YIELDS_NULL OFF 
GO
ALTER DATABASE [dataset] SET NUMERIC_ROUNDABORT OFF 
GO
ALTER DATABASE [dataset] SET QUOTED_IDENTIFIER OFF 
GO
ALTER DATABASE [dataset] SET RECURSIVE_TRIGGERS OFF 
GO
ALTER DATABASE [dataset] SET  DISABLE_BROKER 
GO
ALTER DATABASE [dataset] SET AUTO_UPDATE_STATISTICS_ASYNC OFF 
GO
ALTER DATABASE [dataset] SET DATE_CORRELATION_OPTIMIZATION OFF 
GO
ALTER DATABASE [dataset] SET TRUSTWORTHY OFF 
GO
ALTER DATABASE [dataset] SET ALLOW_SNAPSHOT_ISOLATION OFF 
GO
ALTER DATABASE [dataset] SET PARAMETERIZATION SIMPLE 
GO
ALTER DATABASE [dataset] SET READ_COMMITTED_SNAPSHOT OFF 
GO
ALTER DATABASE [dataset] SET HONOR_BROKER_PRIORITY OFF 
GO
ALTER DATABASE [dataset] SET RECOVERY FULL 
GO
ALTER DATABASE [dataset] SET  MULTI_USER 
GO
ALTER DATABASE [dataset] SET PAGE_VERIFY CHECKSUM  
GO
ALTER DATABASE [dataset] SET DB_CHAINING OFF 
GO
ALTER DATABASE [dataset] SET FILESTREAM( NON_TRANSACTED_ACCESS = OFF ) 
GO
ALTER DATABASE [dataset] SET TARGET_RECOVERY_TIME = 60 SECONDS 
GO
ALTER DATABASE [dataset] SET DELAYED_DURABILITY = DISABLED 
GO
ALTER DATABASE [dataset] SET ACCELERATED_DATABASE_RECOVERY = OFF  
GO
EXEC sys.sp_db_vardecimal_storage_format N'dataset', N'ON'
GO
ALTER DATABASE [dataset] SET QUERY_STORE = ON
GO
ALTER DATABASE [dataset] SET QUERY_STORE (OPERATION_MODE = READ_WRITE, CLEANUP_POLICY = (STALE_QUERY_THRESHOLD_DAYS = 30), DATA_FLUSH_INTERVAL_SECONDS = 900, INTERVAL_LENGTH_MINUTES = 60, MAX_STORAGE_SIZE_MB = 1000, QUERY_CAPTURE_MODE = AUTO, SIZE_BASED_CLEANUP_MODE = AUTO, MAX_PLANS_PER_QUERY = 200, WAIT_STATS_CAPTURE_MODE = ON)
GO
USE [dataset]
GO
/****** Object:  Table [dbo].[dataset]    Script Date: 31/07/2025 11:02:10 ******/
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
CREATE TABLE [dbo].[dataset](
	[id] [int] IDENTITY(1,1) NOT NULL,
	[image_name] [nvarchar](255) NULL,
	[chemin_image] [nvarchar](500) NULL,
	[infection_class] [nvarchar](50) NULL,
	[infection_percent] [float] NULL,
	[confiance] [float] NULL,
	[decision] [nvarchar](50) NULL,
PRIMARY KEY CLUSTERED 
(
	[id] ASC
)WITH (PAD_INDEX = OFF, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON, OPTIMIZE_FOR_SEQUENTIAL_KEY = OFF) ON [PRIMARY]
) ON [PRIMARY]
GO
USE [master]
GO
ALTER DATABASE [dataset] SET  READ_WRITE 
GO
