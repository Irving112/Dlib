from demo import get_register_features, compare_face_fatures_with_database, White_plot

if __name__ == '__main__':
    RegisterPath = r'F:\face_automation\material\Register'
    RecogizePath = r'F:\face_automation\material\Recognize\test'
    #获取注册照人脸特征值
    get_register_features(RegisterPath)
    #注册照识别照匹配
    compare_face_fatures_with_database(RecogizePath)
    #输出人脸框并写入json
    White_plot()