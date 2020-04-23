workspace "ai-playground"
    architecture "x64"
    configurations
    {
        "debug",
        "release"
    }

    outputdir = "%{cfg.buildcfg}"

project "playground"
    kind "ConsoleApp"
    language "C++"

    targetdir ("bin/" .. outputdir .. "/")
    objdir ("bin-int/" .. outputdir .. "/")

    files
    {
        "src/**.h",
        "src/**.cpp"
    }

    includedirs
    {
         "src/",
         "vendor/include/"
    }

    links
    {
    }

    filter "system:windows"
        cppdialect "C++17"
        staticruntime "On"
        systemversion "latest"

        defines
        {
            "OS_WINDOWS"
        }

    filter "system:macosx"
        cppdialect "C++17"
        staticruntime "On"
        systemversion "latest"

        defines
        {
            "OS_MACOS"
        }

    filter "system:linux"
        cppdialect "C++17"
        staticruntime "On"
        systemversion "latest"

        defines
        {
            "OS_LINUX"
        }
        
    filter "configurations:debug"
        defines { "DEBUG" }
        symbols "On"
    

    filter "configurations:release"
        defines { "NDEBUG" }
        optimize "On"
