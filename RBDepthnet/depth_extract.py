import sys
import os
import datetime
import renderdoc as rd

'''
extract depth Image from renderdoc
'''

#save src image
def save_src(action, controller, destPath, preName):
    textureSave = rd.TextureSave()
    textureSave.resourceId = action.outputs[0]
    textureSave.alpha = rd.AlphaMapping.Discard
    textureSave.mip = 0
    textureSave.slice.sliceIndex = 0
    textureSave.jpegQuality = 100
    textureSave.destType = rd.FileType.JPG
    controller.SaveTexture(textureSave, os.path.join(destPath,preName+'_1.jpg'))
#save depth image
def save_depth(action, controller, destPath, preName, flag):
    textureSave = rd.TextureSave()
    textureSave.resourceId = action.depthOut
    textureSave.alpha = rd.AlphaMapping.Preserve
    textureSave.mip = -1
    textureSave.destType = rd.FileType.PNG
    if flag==True:
        controller.SaveTexture(textureSave, os.path.join(destPath,preName+'_2.png'))
    else:
        controller.SaveTexture(textureSave, os.path.join(destPath,preName+'_'+str(action.eventId)+'.png'))
#load rdcfile
def loadCapture(filename):
    cap = rd.OpenCaptureFile()
    status = cap.OpenFile(filename, '', None)
    # Make sure the file opened successfully
    if status != rd.ReplayStatus.Succeeded:
        raise RuntimeError("Couldn't open file: " + str(status))

    # Make sure we can replay
    if not cap.LocalReplaySupport():
        raise RuntimeError("Capture cannot be replayed")

    # Initialise the replay
    status, controller = cap.OpenCapture(rd.ReplayOptions(), None)

    if status != rd.ReplayStatus.Succeeded:
        raise RuntimeError("Couldn't initialise replay: " + str(status))

    return (cap, controller)
#main
def main(rdcPath, destPath, srcId, depthId):
    rdcfiles = os.listdir(rdcPath)
    time = datetime.date.today()
    index = 1
    for file in rdcfiles:
        filename = os.path.join(rdcPath,file)
        filesize = os.path.getsize(filename)/float(1024*1024)
        if filesize < 70: #skip file less than 70MB
            continue
        print(filename)
        cap, controller = loadCapture(filename)
        for action in controller.GetRootActions():
            print('action.eventId:', action.eventId)
            # print(int(action.outputs[0]))
            controller.SetFrameEvent(action.eventId, True)
            if int(action.outputs[0])==srcId and int(action.depthOut)==depthId:
                next_action = action.next
                next_action_name = next_action.GetName(controller.GetStructuredFile()).split('::')[1]
                preName = str(time) + '_' + str(index)
                if next_action_name=='Draw()' or next_action_name=='DiscardView()':
                    save_src(action, controller, destPath, preName)
                    save_depth(action, controller, destPath, preName, flag=True)
                    break
                else:
                    save_depth(action, controller, destPath, preName, flag=False)
        controller.Shutdown()
        cap.Shutdown()
        rd.ShutdownReplay()
        index += 1

path = 'E:\\gametodata-master\\12-12-rdc-1' # path of rdc file list
dest_path = 'E:\\game2data_FIFA\\12-12'
if not os.path.exists(dest_path):
    os.makedirs(dest_path)
src_id = 4358
depth_id = 4361
main(path,dest_path,src_id,depth_id)