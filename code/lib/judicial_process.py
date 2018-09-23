# -*- coding: utf-8 -*-
#
# ['被执行人' '被告' '上诉人(原审被告)' '上诉人' '一审被告' '被申请人' '原审被告' '被上诉人' '原告' '申请执行人'
#  '上诉人(原审原告)' '申请人' '被上诉人(原审原告)' '申请再审人' '被告单位' '原告(反诉被告)' '原审第三人' '第三人'
#  '再审申请人' '案外人' '被上诉人(原审被告)' '起诉人' '异议人' '上诉人(原审原告、反诉被告)' '异议人（被执行人）'
#  '复议申请人' '反诉被告' '被审申请人' '被上诉人(原审原告、反诉被告)' '被申请人(一审原告)' '再审被申请人'
#  '再审申请人（一审原告、二审上诉人）' '原审原告' '上诉人(原审被告、反诉原告)' '复议申请人（申请执行人）' '被申请执行人'
#  # '被罚款人' '被申诉人' '被告人' '原审被告人' '被申请人(一审被告、二审被上诉人)' '被告(反诉原告)']

# [
# '原告' '原告/上诉人' '上诉人'  '公诉人/原告/上诉人/申请人' '当事人/公诉人'

# '被告'  '被告人' '被告/被上诉人' '被告/被告人/被上诉人/被申请人' '原审被告' '原审被告)'

# '第三人''原审第三人'

# '申请人' '申请人:被告' '申请人:原告'  '申请人:第三人' '申请人:上诉人'

#  '被申请人'

# '上诉人原告' '上诉人被告'
# '被上诉人'   '被上诉人原告' '被上诉人被告'


# '当事人'
'被执行人'

'被申请执行人'
'被申请人(原审被告)' '被申请人(原审原告)'

# '未知'


#  '申请执行人'
'债务人'
'债权人'
'申请再审人'
'被申请再审人'
'公告内容'
'申诉人'
'异议人(申请执行人)'  '原审(一审)'
#
#     ]

def adjudicative_documents_fole(text):
    if text.find('原告') !=  -1 or text == '原告' or text == '原审原告':
        return 1
    if text.find('被告') != -1 and text.find('上诉人') == -1 :
        return 2
    if text.find('上诉人') != -1 and text.find('申请人') == -1 :
        return 3
    if text == '申请人':
        return 4
    if text.find('被申请人') != -1 and text != '再审被申请人':
        return 5
    if text == '申请执行人':
        return 6
    if text == '被申请执行人':
        return 7
    if text == '申请再审人' or text.find('再审申请人') != -1:
        return 8
    if text == '再审被申请人':
        return 9
    if text == '起诉人':
        return 10
    if text == '被执行人':
        return 11
    if text.find('第三人') != -1:
        return 12
    if text.find('案外人') != -1:
        return 13
    if text.find('异议人') != -1:
        return 14
    if text.find('复议申请人') != -1:
        return 15
    if text == '被审申请人':
        return 16
    if text == '被罚款人':
        return 17
    if text == '被申诉人':
        return 18
    return 0


# ['江苏省常熟市人民法院' '江苏省南通市中级人民法院' '江苏省高级人民法院' '江苏省苏州市虎丘区人民法院' '江苏省苏州工业园区人民法院'
#  '江苏省苏州市吴中区人民法院' '江苏省苏州市中级人民法院' '江苏省苏州市相城区人民法院' '安徽省高级人民法院' '安徽省宣城市中级人民法院'
#  '江苏省昆山市人民法院' '江苏省张家港市人民法院' '江苏省太仓市人民法院' '江苏省常州市天宁区人民法院' '江苏省江阴市人民法院'
#  '江苏省淮安市中级人民法院' '江苏省苏州市吴江区人民法院' '江苏省淮安市清浦区人民法院' '上海市青浦区人民法院'
#  '江苏省南京市江宁区人民法院' '上海市嘉定区人民法院' '江苏省无锡市中级人民法院' '上海市奉贤区人民法院' '浙江省宁波市江东区人民法院'
#  '广东省中山市第二人民法院' '山东省青岛市中级人民法院' '浙江省嘉兴市南湖区人民法院' '北京市西城区人民法院' '北京市高级人民法院'
#  '北京知识产权法院' '浙江省江山市人民法院' '江苏省苏州市姑苏区人民法院' '广东省广州市天河区人民法院' '上海市浦东新区人民法院'
#  '安徽省芜湖市弋江区人民法院' '江西省瑞昌市人民法院' '福建省南靖县人民法院' '江西省安福县人民法院' '贵州省遵义市汇川区人民法院'
#  '江苏省靖江市人民法院' '浙江省慈溪市人民法院' '江苏省海安县人民法院' '陕西省西安市未央区人民法院' '山东省惠民县人民法院'
#  '山东省青州市人民法院' '上海市第二中级人民法院' '安徽省铜陵市义安区人民法院' '安徽省铜陵县人民法院'
#  '江苏省无锡高新技术产业开发区人民法院' '江苏省无锡市崇安区人民法院' '重庆市九龙坡区人民法院' '广东省深圳市中级人民法院'
#  '广西壮族自治区宾阳县人民法院' '江苏省淮安经济技术开发区人民法院' '广东省鹤山市人民法院' '海南省海口市中级人民法院'
#  '广东省珠海市中级人民法院' '海南省海口市秀英区人民法院' '广东省珠海市斗门区人民法院' '浙江省杭州市上城区人民法院'
#  '江苏省常州市新北区人民法院' '江苏省泰州市中级人民法院' '山东省莱芜市中级人民法院' '重庆市渝中区人民法院'
#  '河南省洛阳市涧西区人民法院' '江苏省常州市中级人民法院' '江苏省无锡市锡山区人民法院' '江苏省连云港市赣榆区人民法院'
#  '辽宁省大连市甘井子区人民法院' '河北省泊头市人民法院' '江苏省无锡市惠山区人民法院' '山西省太原市万柏林区人民法院'
#  '上海市金山区人民法院' '重庆市南岸区人民法院' '江苏省南京市溧水区人民法院' '浙江省杭州市余杭区人民法院' '江苏省泰州市高港区人民法院'
#  '江苏省南京市六合区人民法院' '江苏省盐城市中级人民法院' '上海海事法院' '广东省深圳市龙岗区人民法院' '江苏省扬州市广陵区人民法院'
#  '浙江省杭州市江干区人民法院' '浙江省安吉县人民法院' '河南省汝州市人民法院' '浙江省绍兴市上虞区人民法院' '江苏省如皋市人民法院'
#  '江苏省南京市鼓楼区人民法院' '上海市闵行区人民法院' '上海市第一中级人民法院' '重庆市长寿区人民法院' '浙江省嘉兴市中级人民法院'
#  '浙江省嘉善县人民法院' '江苏省南通市港闸区人民法院' '湖北省宜昌市猇亭区人民法院' '安徽省合肥市瑶海区人民法院'
#  '福建省厦门市集美区人民法院' '湖北省武汉市中级人民法院' '湖北省武汉市汉南区人民法院' '广东省佛山市三水区人民法院'
#  '江苏省常州市钟楼区人民法院' '浙江省桐乡市人民法院' '浙江省象山县人民法院' '上海市松江区人民法院' '上海市徐汇区人民法院'
#  '江苏省无锡市新吴区人民法院' '甘肃省酒泉市中级人民法院' '江苏省淮安市淮安区人民法院' '江苏省南京市中级人民法院'
#  '北京市第二中级人民法院' '河北省高级人民法院' '河北省张家口市中级人民法院' '山东省青岛市城阳区人民法院' '上海知识产权法院'
#  '山东省济阳县人民法院' '江苏省徐州市中级人民法院' '山东省寿光市人民法院' '上海市宝山区人民法院' '广东省珠海市香洲区人民法院'
#  '安徽省马鞍山市中级人民法院' '安徽省马鞍山市雨山区人民法院' '浙江省宁波市北仑区人民法院' '江苏省常州市武进区人民法院'
#  '陕西省宝鸡市渭滨区人民法院' '山东省龙口市人民法院' '山东省胶州市人民法院' '广东省东莞市中级人民法院' '江苏省南京市浦口区人民法院'
#  '浙江省瑞安市人民法院' '浙江省温州市鹿城区人民法院' '浙江省温州市中级人民法院' '浙江省温州市瓯海区人民法院' '江苏省宝应县人民法院'
#  '河南省泌阳县人民法院' '安徽省芜湖市镜湖区人民法院' '浙江省玉环市人民法院' '浙江省宁波市镇海区人民法院' '广东省广州市白云区人民法院'
#  '浙江省台州市黄岩区人民法院' '安徽省合肥高新技术产业开发区人民法院' '浙江省新昌县人民法院' '广西壮族自治区柳州市鱼峰区人民法院'
#  '山东省青岛市黄岛区人民法院' '江苏省盐城市亭湖区人民法院' '山东省德州市德城区人民法院' '广东省深圳市福田区人民法院'
#  '江苏省无锡市滨湖区人民法院' '湖北省武汉市江汉区人民法院' '广东省深圳市罗湖区人民法院' '江苏省丹阳市人民法院' '江苏省涟水县人民法院'
#  '江苏省泰州市姜堰区人民法院' '江苏省扬州市江都区人民法院' '浙江省绍兴县人民法院' '四川省成都市锦江区人民法院'
#  '山东省淄博市周村区人民法院' '江苏省淮安市淮阴区人民法院' '安徽省池州市贵池区人民法院' '浙江省文成县人民法院' '江苏省句容市人民法院'
#  '浙江省杭州市萧山区人民法院' '河南省高级人民法院' '浙江省宁波市鄞州区人民法院' '浙江省海盐县人民法院' '浙江省余姚市人民法院'
#  '上海市静安区人民法院' '福建省高级人民法院' '江苏省扬中市人民法院' '新疆维吾尔自治区昌吉回族自治州中级人民法院'
#  '浙江省义乌市人民法院' '浙江省宁波市江北区人民法院' '江苏省南通经济技术开发区人民法院' '浙江省海宁市人民法院'
#  '安徽省滁州市琅琊区人民法院' '上海市虹口区人民法院' '辽宁省沈阳市大东区人民法院' '浙江省湖州市吴兴区人民法院'
#  '浙江省绍兴市柯桥区人民法院' '四川省南充市顺庆区人民法院' '北京市第一中级人民法院' '河北省怀安县人民法院'
#  '浙江省嘉兴市秀洲区人民法院' '杭州铁路运输法院' '广东省深圳市宝安区人民法院' '最高人民法院' '山东省烟台经济技术开发区人民法院'
#  '江苏省宜兴市人民法院' '上海市高级人民法院' '上海市杨浦区人民法院' '江苏省泰州市海陵区人民法院' '江苏省南通市崇川区人民法院'
#  '山东省烟台市芝罘区人民法院' '安徽省宣城市宣州区人民法院' '山东省桓台县人民法院' '江西省上饶市广丰区人民法院' '江苏省射阳县人民法院'
#  '上海市普陀区人民法院' '广东省东莞市第二人民法院' '湖北省襄阳市樊城区人民法院' '江西省赣州市章贡区人民法院'
#  '福建省厦门市中级人民法院' '辽宁省沈阳市中级人民法院' '云南省玉溪市中级人民法院' '北京市昌平区人民法院' '重庆市渝北区人民法院'
#  '山东省威海市环翠区人民法院' '江苏省盱眙县人民法院' '内蒙古自治区呼和浩特市中级人民法院' '湖北省枝江市人民法院'
#  '四川省成都市武侯区人民法院' '山东省诸城市人民法院' '江苏省南京市秦淮区人民法院' '江苏省宿迁市中级人民法院' '江苏省建湖县人民法院'
#  '江苏省沛县人民法院' '吉林省长春市中级人民法院' '吉林省长春市二道区人民法院' '北京市第三中级人民法院' '浙江省德清县人民法院'
#  '浙江省平湖市人民法院' '江苏省睢宁县人民法院' '辽宁省大连市中级人民法院' '大连海事法院' '天津市滨海新区人民法院'
#  '福建省惠安县人民法院' '重庆市忠县人民法院' '浙江省宁波市中级人民法院' '浙江省永康市人民法院' '安徽省黄山市徽州区人民法院'
#  '安徽省蚌埠市禹会区人民法院' '安徽省芜湖经济技术开发区人民法院' '河南省原阳县人民法院' '福建省福州市中级人民法院'
#  '福建省连江县人民法院' '浙江省嵊州市人民法院' '江苏省阜宁县人民法院' '辽宁省大连市金州区人民法院' '江苏省扬州市中级人民法院'
#  '江苏省泰兴市人民法院' '江苏省金坛市人民法院' '江苏省镇江市京口区人民法院' '江苏省仪征市人民法院' '湖北省咸宁市中级人民法院'
#  '湖北省通城县人民法院' '安徽省安庆市中级人民法院' '浙江省湖州市南浔区人民法院' '安徽省滁州市南谯区人民法院'
#  '广东省广州市中级人民法院' '浙江省杭州市滨江区人民法院' '浙江省丽水市莲都区人民法院' '江苏省南京市雨花台区人民法院'
#  '广东省佛山市中级人民法院' '武汉海事法院' '浙江省天台县人民法院' '江西省南昌市青山湖区人民法院' '湖北省广水市人民法院'
#  '江苏省无锡市南长区人民法院' '山东省淄博高新技术产业开发区人民法院' '江苏省吴江市人民法院' '江苏省东台市人民法院'
#  '江苏省丰县人民法院' '重庆市第一中级人民法院' '江苏省南京市建邺区人民法院' '北京市朝阳区人民法院' '辽宁省沈阳经济技术开发区人民法院'
#  '山东省高级人民法院' '广东省肇庆市中级人民法院' '安徽省宿州市埇桥区人民法院' '江苏省滨海县人民法院' '浙江省金华市中级人民法院'
#  '浙江省衢州市中级人民法院' '天津市北辰区人民法院' '安徽省芜湖县人民法院' '浙江省平阳县人民法院' '江苏省海门市人民法院'
#  '安徽省当涂县人民法院' '山东省沂源县人民法院' '安徽省铜陵市狮子山区人民法院' '江苏省高淳县人民法院' '安徽省桐城市人民法院'
#  '江苏省启东市人民法院' '江西省萍乡市中级人民法院' '江西省莲花县人民法院' '浙江省台州市路桥区人民法院' '江苏省连云港市海州区人民法院'
#  '江苏省连云港市中级人民法院' '山东省临朐县人民法院' '江苏省盐城市盐都区人民法院' '广东省广州市从化区人民法院' '上海市长宁区人民法院'
#  '江苏省镇江经济开发区人民法院' '江苏省南京市玄武区人民法院' '浙江省台州市中级人民法院' '浙江省绍兴市中级人民法院'
#  '江苏省沭阳县人民法院' '浙江省杭州市西湖区人民法院' '江苏省泗阳县人民法院' '湖南省长沙市中级人民法院' '安徽省铜陵市铜官山区人民法院'
#  '上海市黄浦区人民法院' '安徽省长丰县人民法院' '浙江省长兴县人民法院' '浙江省诸暨市人民法院' '广东省惠州市惠城区人民法院'
#  '天津市第一中级人民法院' '浙江省温岭市人民法院' '广东省东莞市第三人民法院' '湖北省沙洋人民法院' '浙江省高级人民法院'
#  '浙江省杭州市中级人民法院' '天津市河西区人民法院' '浙江省温州市龙湾区人民法院' '江苏省宿迁市宿城区人民法院'
#  '湖北省鄂州市鄂城区人民法院' '福建省厦门市湖里区人民法院' '福建省长泰县人民法院' '重庆市合川区人民法院' '湖南省宁乡县人民法院'
#  '四川省成都高新技术产业开发区人民法院' '河南省安阳市中级人民法院' '河南省安阳县人民法院' '江苏省淮安市清江浦区人民法院'
#  '河南省南阳市中级人民法院' '河南省淅川县人民法院' '山东省济宁高新技术产业开发区人民法院' '安徽省池州市中级人民法院'
#  '安徽省宿州市中级人民法院' '山东省烟台市莱山区人民法院' '河南省栾川县人民法院' '浙江省杭州市下城区人民法院'
#  '浙江省绍兴市越城区人民法院' '山西省泽州县人民法院' '山西省阳泉市郊区人民法院' '安徽省广德县人民法院' '天津市南开区人民法院'
#  '山东省章丘市人民法院' '四川省绵阳市游仙区人民法院' '四川省绵阳市中级人民法院' '湖南省益阳市中级人民法院' '浙江省玉环县人民法院'
#  '安徽省肥西县人民法院' '江苏省淮安市清河区人民法院' '江苏省无锡市北塘区人民法院' '安徽省太湖县人民法院'
#  '新疆维吾尔自治区阿克苏地区中级人民法院' '安徽省歙县人民法院' '江苏省高邮市人民法院' '江苏省扬州市邗江区人民法院'
#  '山东省淄博市博山区人民法院' '安徽省蚌埠市龙子湖区人民法院' '安徽省蚌埠市中级人民法院' '山东省济南市历下区人民法院'
#  '江苏省金湖县人民法院' '江苏省连云港市连云区人民法院' '四川省夹江县人民法院' '安徽省郎溪县人民法院' '浙江省杭州市拱墅区人民法院'
#  '河南省正阳县人民法院' '江西省金溪县人民法院' '浙江省乐清市人民法院' '河南省新密市人民法院' '江苏省无锡市梁溪区人民法院'
#  '福建省漳平市人民法院' '江苏省南通市通州区人民法院' '浙江省宁波市海曙区人民法院' '浙江省宁海县人民法院' '河南省郑州市惠济区人民法院'
#  '安徽省繁昌县人民法院' '陕西省渭南市临渭区人民法院' '江西省上栗县人民法院' '贵州省铜仁市中级人民法院' '四川省南充市嘉陵区人民法院'
#  '贵州省德江县人民法院' '贵州省习水县人民法院' '湖北省孝感市中级人民法院' '山东省德州市中级人民法院' '北京市大兴区人民法院'
#  '河南省郑州市中级人民法院' '河南省郑州高新技术产业开发区人民法院' '湖北省随州市中级人民法院' '浙江省兰溪市人民法院'
#  '广东省惠州市中级人民法院' '广东省博罗县人民法院' '新疆生产建设兵团第一师中级人民法院' '湖南省长沙市望城区人民法院'
#  '黑龙江省大庆高新技术产业开发区人民法院' '新疆维吾尔自治区乌鲁木齐市头屯河区人民法院' '河南省平顶山市中级人民法院' '河南省郏县人民法院'
#  '江苏省徐州经济技术开发区人民法院' '重庆市江北区人民法院' '江西省抚州市临川区人民法院' '山东省滨州市滨城区人民法院'
#  '安徽省明光市人民法院' '湖北省通山县人民法院' '江苏省常州市金坛区人民法院' '山东省临沂市兰山区人民法院' '北京市密云区人民法院'
#  '广东省东莞市第一人民法院' '北京市通州区人民法院' '江苏省响水县人民法院' '山东省威海火炬高技术产业开发区人民法院'
#  '山东省即墨市人民法院' '湖南省常德市武陵区人民法院' '江苏省盐城市大丰区人民法院' '河南省濮阳县人民法院' '湖南省澧县人民法院'
#  '广东省珠海横琴新区人民法院' '福建省厦门市海沧区人民法院' '天津市第二中级人民法院' '浙江省富阳市人民法院' '江苏省镇江市中级人民法院'
#  '辽宁省沈阳市和平区人民法院' '福建省泉州市丰泽区人民法院' '四川省绵阳市涪城区人民法院' '山西省太原市迎泽区人民法院'
#  '江苏省江都市人民法院' '陕西省西安市长安区人民法院' '河北省石家庄市中级人民法院' '福建省上杭县人民法院' '福建省龙岩市新罗区人民法院'
#  '安徽省枞阳县人民法院' '浙江省奉化市人民法院' '四川省乐山市市中区人民法院' '湖北省宜昌市伍家岗区人民法院' '江苏省溧阳市人民法院'
#  '广东省江门市中级人民法院' '四川省泸州市江阳区人民法院']

def adjudicative_documents_institution(text):
    if text.find('市人民法院') != -1 or text.find('市第二人民法院') != -1 or text.find('市第三人民法院') != -1:
        return 1
    if text.find('中级人民法院') != -1:
        return 2
    if text.find('高级人民法院') != -1:
        return 3
    if text.find('区人民法院') != -1 or text.find('县人民法院') != -1 or text.find('沙洋人民法院') != -1:
        return 4
    if text.find('知识产权法院') != -1:
        return 5
    if text.find('海事法院') != -1:
        return 6
    if text.find('铁路运输法院') != -1:
        return 7
    if text.find('最高人民法院') != -1:
        return 8
    return 0


# ['执行实施类案件' '债权转让合同纠纷' '民间借贷纠纷' '金融借款合同纠纷' '追偿权纠纷' '买卖合同纠纷' '担保追偿权纠纷'
#  '借款合同纠纷' '保证合同纠纷' '股东资格确认纠纷' '股东知情权纠纷' '劳动争议' '房屋租赁合同纠纷' '追索劳动报酬纠纷'
#  '合同纠纷' '装饰装修合同纠纷' '执行异议' '分期付款买卖合同纠纷' '侵犯技术秘密纠纷' '建设工程施工合同纠纷' '小额借款合同纠纷'
#  '申请诉前财产保全' '债权人撤销权纠纷' '申请公示催告' '加工合同纠纷' '商品房预售合同纠纷' '确认劳动关系纠纷' '融资租赁合同纠纷'
#  '定作合同纠纷' '生命权、健康权、身体权纠纷' '广告合同纠纷' ' 债权纠纷' '商品房销售合同纠纷' '居间合同纠纷' '社会保险纠纷'
#  '服务合同纠纷' '侵害商标权纠纷' '委托合同纠纷' '提供劳务者受害责任纠纷' '无因管理纠纷' '确认票据无效纠纷' '车辆租赁合同纠纷'
#  '非诉行政行为申请执行审查案件' '承揽合同纠纷' '侵犯外观设计专利权纠纷' '专利权权属、侵权纠纷' '餐饮服务合同纠纷' '侵权责任纠纷'
#  '运输合同纠纷' '虚开增值税专用发票、用于骗取出口退税、抵扣税款发票罪' '著作权权属、侵权纠纷' '土地租赁合同纠纷' '侵犯著作财产权纠纷'
#  '租赁合同纠纷' '垄断纠纷' '房屋买卖合同纠纷' '机动车交通事故责任纠纷' '工伤保险待遇纠纷' '公路货物运输合同纠纷'
#  '一般人格权纠纷' '企业借贷纠纷' '恢复执行案件' '技术服务合同纠纷' '侵犯实用新型专利权纠纷' '物权确认纠纷' '财产损害赔偿纠纷'
#  '城乡建设行政管理' '公安行政管理' '票据纠纷' '债权人代位权纠纷' '申请破产清算' '经济补偿金纠纷' '票据追索权纠纷'
#  '申请支付令' '企业租赁经营合同纠纷' '商品房委托代理销售合同纠纷' '执行异议案件' '公司证照返还纠纷' '公司决议效力确认纠纷'
#  '劳动和社会保障行政管理' '股权转让纠纷' '返还原物纠纷' '建设用地使用权出让合同纠纷' '土地使用权转让合同纠纷'
#  '侵害作品信息网络传播权纠纷' '海上、通海水域货物运输合同纠纷' '人事争议' '未知' '公司清算纠纷' '公司解散纠纷' '名誉权纠纷'
#  '邮寄服务合同纠纷' '供用水合同纠纷' '建设工程合同纠纷' '请求变更公司登记纠纷' '招标投标买卖合同纠纷' '产品销售者责任纠纷'
#  '企业承包经营合同纠纷' '票据损害责任纠纷' '仓储合同纠纷' '劳动合同纠纷' '执行查封拍卖抵债' '债务转移合同纠纷'
#  '道路交通事故人身损害赔偿纠纷' '修理合同纠纷' '商品房预约合同纠纷' '申请确认仲裁协议效力' '执行异议之诉' '所有权确认纠纷'
#  '票据付款请求权纠纷' '专利权权属纠纷' '普通破产债权确认纠纷' '建设工程分包合同纠纷' '道路交通管理' '财产保全执行案件'
#  '缔约过失责任纠纷' '票据返还请求权纠纷' '侵害其他著作财产权纠纷' '劳务派遣合同纠纷' '侵害作品表演权纠纷' '申请撤销仲裁裁决'
#  '行政确认' '金融不良债权追偿纠纷' '专利代理合同纠纷' '不当得利纠纷' '商标行政管理' '票据利益返还请求权纠纷'
#  '产品质量损害赔偿纠纷' '损害公司利益责任纠纷' '工商行政管理' '货运代理合同纠纷' '劳务合同纠纷' '保险合同纠纷' '虚假宣传纠纷'
#  '票据交付请求权纠纷' '网络购物合同纠纷' '股东出资纠纷' '定金合同纠纷' '物业服务合同纠纷' '执行复议案件' '确认合同效力纠纷'
#  '人格权纠纷' '侵犯发明专利权纠纷' '旅店服务合同纠纷' '其它民事纠纷' '侵害作品放映权纠纷' '土地行政管理' '损害股东利益责任纠纷'
#  '肖像权纠纷' '因申请诉中财产保全损害责任纠纷' '财产保全及其解除' '恢复原状纠纷' '产品责任纠纷' '侵犯商业秘密纠纷'
#  '确认合同无效纠纷' '典当纠纷' '城市规划管理' '占有物返还纠纷' '案外人执行异议之诉' '因申请诉前财产保全损害责任纠纷'
#  '建设工程监理合同纠纷' '知识产权合同纠纷' '侵害作品发表权纠纷' '特许经营合同纠纷' '专利行政管理' '房屋拆迁安置补偿合同纠纷'
#  '教育机构责任纠纷' '财产保险合同纠纷' '侵犯著作人身权纠纷' '房地产咨询合同纠纷' '教育培训合同纠纷' '合伙协议纠纷'
#  '财产损失保险合同纠纷' '与破产有关的纠纷' '第三人撤销之诉' '取回权纠纷' '航空货物运输合同纠纷' '挂靠经营合同纠纷'
#  '水污染侵权纠纷' '技术合同纠纷' '计算机软件开发合同纠纷' '信用证纠纷' '民间委托理财合同纠纷' '进出口代理合同纠纷'
#  '仲裁程序中的财产保全' '请求确认人民调解协议效力' '物价行政管理' '船舶物料和备品供应合同纠纷' '保险人代位求偿权纠纷'
#  '侵害作品复制权纠纷' '信息网络传播权纠纷' '商标异议' '公司决议纠纷' '管辖异议' '车位纠纷' '车库纠纷' '走私普通货物、物品罪'
#  '建筑设备租赁合同纠纷' '国际货物买卖合同纠纷' '请求撤销个别清偿行为纠纷' '船舶买卖合同纠纷' '离退休人员返聘合同纠纷'
#  '普通合伙纠纷' '辞退争议' '建设工程设计合同纠纷' '建设用地使用权纠纷' '排除妨害纠纷' '信用证欺诈纠纷' '破产撤销权纠纷'
#  '司法罚款案件' '股东损害公司债权人利益责任纠纷' '网络服务合同纠纷' '确认合同有效纠纷' '义务帮工人受害责任纠纷' '侵害技术秘密纠纷'
#  '不正当竞争纠纷' '危险驾驶罪' '故意伤害罪' '修理、重作、更换纠纷' '著作权权属纠纷' '清算责任纠纷' '假冒注册商标罪'
#  '行纪合同纠纷' '保安服务合同纠纷' '税务行政管理' '电信服务合同纠纷' '建设用地使用权合同纠纷' '地面施工、地下设施损害责任纠纷'
#  '债权债务概括转移合同纠纷' '占有保护纠纷' '商标合同纠纷' '公司增资纠纷' '环境保护行政管理' '走私废物罪' '出租汽车运输合同纠纷'
#  '质量监督行政管理' '申请破产重整' '网络域名权属纠纷' '公司盈余分配纠纷' '旅游合同纠纷' '计算机软件著作权许可使用合同纠纷'
#  '商业诋毁纠纷' '法律服务合同纠纷' '侵犯商标专用权纠纷' '实现担保物权' '联营合同纠纷' '职务侵占罪' '消费者权益保护纠纷'
#  '意外伤害保险合同纠纷' '合资、合作开发房地产合同纠纷' '确认不侵害专利权纠纷' '保险纠纷' '借用合同纠纷' '技术咨询合同纠纷'
#  '房屋拆迁管理' '海上、通海水域保险合同纠纷']