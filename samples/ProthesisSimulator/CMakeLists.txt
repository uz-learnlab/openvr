set(TARGET_NAME prothesis_simulator)

find_package( OpenCV REQUIRED )

if(APPLE)
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set_source_files_properties(../shared/pathtools.cpp hellovr_opengl_main.cpp PROPERTIES COMPILE_FLAGS "-x objective-c++")
    find_library(FOUNDATION_FRAMEWORK Foundation)
    mark_as_advanced(FOUNDATION_FRAMEWORK)
    set(EXTRA_LIBS ${EXTRA_LIBS} ${FOUNDATION_FRAMEWORK})
  endif()
endif()

add_executable(${TARGET_NAME}
  ${SHARED_SRC_FILES}
  prothesis_simulator_main.cpp
)

target_link_libraries(${TARGET_NAME}
  ${OPENGL_LIBRARIES}
  ${OpenCV_LIBS}
  ${GLEW_LIBRARIES}
  ${SDL2_LIBRARIES}
  ${OPENVR_LIBRARIES}
  ${CMAKE_DL_LIBS}
  ${EXTRA_LIBS}
)

setTargetOutputDirectory(${TARGET_NAME})
